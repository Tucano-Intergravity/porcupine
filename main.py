#!/usr/bin/env python3
"""
Porcupine 음성 챗봇 (WM8960 + ST7789V LCD)

- porcupine.py 구조 기반
- 음성 입출력: WM8960 HAT (SpeakerLCDTest 참조)
- UI: ST7789V LCD (wm8960_lcd_integrated_test / lcd_test 참조)
- 카메라/OpenCV 창 없음 (헤드리스·LCD 전용)
"""

import os
import sys
import json
import math
import time
import wave
import select
import threading
import subprocess
import numpy as np
import pyaudio
import pvporcupine
from openai import OpenAI
from tavily import TavilyClient
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional

# LCD/GPIO (Raspberry Pi 전용, 선택적)
try:
    import spidev
    import RPi.GPIO as GPIO
    LCD_AVAILABLE = True
except ImportError:
    LCD_AVAILABLE = False

# --- 스크립트 기준 경로 (상위 pi_ai에서 API 키 등 로드) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
os.chdir(PARENT_DIR)

# --- [설정] API 키 로드 ---
def load_api_key(filename: str, default: str = "") -> str:
    """파일에서 API 키를 읽거나 기본값 반환"""
    try:
        for base in [os.getcwd(), SCRIPT_DIR, PARENT_DIR]:
            filepath = os.path.join(base, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    key = f.read().strip().replace("\n", "").replace("\r", "").replace(" ", "")
                    if key and key != "..." and len(key) > 10:
                        print(f"✅ {filename}에서 키 로드 성공 (길이: {len(key)})")
                        return key
        print(f"⚠️ 파일을 찾을 수 없습니다: {filename}")
    except Exception as e:
        print(f"⚠️ 키 파일 읽기 오류 ({filename}): {e}")
    return default

OPENAI_API_KEY = load_api_key("OpenAI_API_Key.txt", "sk-...")
TAVILY_API_KEY = load_api_key("tavily_API_key.txt", "tvly-...")
PICOVOICE_KEY = load_api_key("picovoice_API_Key.txt", "...")

# 키워드 인식 조정 (필요 시 수정)
PORCUPINE_SENSITIVITY = 0.8    # 0~1, 제조사 권장 기본값 0.5 → 인식률 개선 위해 0.8
MIC_SOFTWARE_GAIN = 2.5        # 마이크 소프트웨어 증폭 (1.0=없음)
# 녹음 전 버퍼 비우기 (이전 데이터가 다음 질문에 섞이는 것 방지)
FLUSH_SEC_AFTER_TRIGGER = 0.25  # 트리거 후 버릴 시간(초). 짧을수록 대화 간격 단축
COOLDOWN_SEC_AFTER_RESPONSE = 0.5  # 응답 끝난 뒤 Porcupine 무시 시간(초). 짧을수록 다음 호출 빠름
RECORD_SEC = 3  # (VAD 미사용 시) 고정 녹음 길이(초). VAD 사용 시에는 참고만
# VAD (에너지 기반): 침묵이면 녹음 중지
MAX_RECORD_SEC = 15              # 최대 질문 길이(초)
VAD_SILENCE_SEC = 3              # 음성 없는 구간이 이 시간 이상이면 녹음 중지
VAD_CHUNK_SEC = 0.2               # 에너지 계산 구간(초)
VAD_THRESHOLD = 900               # 구간 평균 에너지가 이 값 넘으면 '말 있음' (0~32768)
# 키워드 인식 시 음성으로 응답할 문장 (TTS 재생 후 녹음 시작)
KEYWORD_ACK_PHRASE = "어 말해봐"
# 노이즈를 음성으로 인식하지 않도록 (0~32768)
MIN_RECORD_LEVEL = 1000           # 전체 평균 레벨이 이 값 미만이면 STT 건너뜀
CHUNK_SEC = 0.5                   # 말 구간 판단용 구간 길이(초)
CHUNK_LEVEL_THRESHOLD = 900       # 구간 평균이 이 값 넘으면 '말 있음' 구간으로 카운트
MIN_CHUNKS_ABOVE_LEVEL = 2        # 이 개수 이상 구간이 임계 넘어야 STT 진행 (노이즈만이면 0~1구간만 넘침)

# 키워드 인식이 잘 안 될 때 예상 원인:
# 1. 마이크 볼륨/감도: amixer 캡처·ADC·LINPUT1 부스트 최대 적용됨. MIC_SOFTWARE_GAIN 올리면 추가 증폭.
# 2. Porcupine 감도: PORCUPINE_SENSITIVITY를 0.9~1.0으로 올리면 인식률 up (오인식도 증가).
# 3. 샘플레이트: Porcupine은 16kHz. WM8960이 16kHz 미지원 시 PyAudio가 리샘플링하거나 오동작할 수 있음.
# 4. 거리/발음: 마이크에 가깝게, "Porcupine"을 뚜렷하게 말하기.
# 5. 주변 소음: 조용한 환경에서 테스트.

# API 키 검사
if PICOVOICE_KEY == "..." or len(PICOVOICE_KEY) < 20:
    print("\n" + "=" * 60)
    print("⚠️  경고: Picovoice API 키가 올바르게 설정되지 않았습니다!")
    print("=" * 60 + "\n")
if OPENAI_API_KEY == "sk-..." or len(OPENAI_API_KEY) < 20:
    print("❌ OpenAI API 키가 올바르게 설정되지 않았습니다!\n")
if TAVILY_API_KEY == "tvly-..." or len(TAVILY_API_KEY) < 10:
    print("⚠️ Tavily API 키가 설정되지 않았습니다.\n")

client = OpenAI(api_key=OPENAI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# --- 전역 상태 ---
shared_state = {
    "text": "시스템 로딩 완료. Porcupine 시동 중...",
    "status": "IDLE",
    "is_running": True,
    "audio_chunk": None,
    "manual_trigger": False,  # True면 키워드 없이 녹음 시작 (Enter 키)
    "cooldown_until": 0.0,    # 이 시간까지 Porcupine 무시 (TTS 에코 방지)
}

LOG_FILE = os.path.join(PARENT_DIR, "jarvis.log")
log_lock = threading.Lock()

def log(msg: str) -> None:
    print(f"\n[JARVIS] {msg}\n")
    with log_lock:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        except Exception:
            pass

# ==================== WM8960 (SpeakerLCDTest 참조) ====================
def get_wm8960_card() -> Optional[int]:
    """시스템에서 WM8960 사운드 카드 번호 반환"""
    try:
        r = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
        for line in r.stdout.split("\n"):
            if "wm8960" in line.lower() and "card" in line.lower():
                return int(line.split("card")[1].split(":")[0].strip())
    except Exception:
        pass
    return None

def _amixer_set(card_id: int, name: str, value: str) -> bool:
    """amixer 설정 실행, 성공 여부 반환"""
    r = subprocess.run(
        ["amixer", "-c", str(card_id), "set", name, value],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return r.returncode == 0


def setup_wm8960_mixer(card_id: int) -> None:
    """WM8960 믹서/경로 설정 + 재생·마이크 감도 최대"""
    c = str(card_id)
    log(f"WM8960(Card {c}) 믹서 설정 중 (재생·마이크 감도 최대)...")

    # --- 출력 경로 ---
    _amixer_set(card_id, "Mono Output Mixer Left", "on")
    _amixer_set(card_id, "Mono Output Mixer Right", "on")

    # --- 재생 볼륨 (스피커 90%) ---
    _amixer_set(card_id, "Speaker", "90%")
    _amixer_set(card_id, "Speaker Playback Volume", "90%")
    _amixer_set(card_id, "Headphone", "100%")
    _amixer_set(card_id, "Headphone Playback Volume", "100%")
    _amixer_set(card_id, "Playback", "100%")
    _amixer_set(card_id, "Playback Volume", "100%")

    # --- 마이크 입력 경로 ---
    _amixer_set(card_id, "Left Boost Mixer LINPUT1", "on")
    _amixer_set(card_id, "Right Boost Mixer RINPUT1", "on")

    # --- 캡처/마이크 감도 최대 (Porcupine 키워드 인식용) ---
    _amixer_set(card_id, "Capture", "100%")
    _amixer_set(card_id, "Capture Volume", "90%")
    _amixer_set(card_id, "ADC PCM Capture Volume", "80%")
    # LINPUT1/RINPUT1 마이크 부스트 (범위 0~3, 3=최대)
    _amixer_set(card_id, "Left Input Boost Mixer LINPUT1 Volume", "3")
    _amixer_set(card_id, "Right Input Boost Mixer RINPUT1 Volume", "3")

    log("WM8960 믹서 설정 완료 (재생·마이크 최대).")

def get_wm8960_pyaudio_index() -> Optional[int]:
    """PyAudio 기본 입력 디바이스 중 WM8960 인덱스 반환"""
    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = (info.get("name") or "").lower()
            if info.get("maxInputChannels", 0) > 0 and "wm8960" in name:
                log(f"WM8960 PyAudio 입력 디바이스: index={i}, name={info.get('name')}")
                return i
    finally:
        pa.terminate()
    return None

TTS_PLAY_TIMEOUT = 120  # TTS 재생 최대 대기(초). 긴 응답은 30초 넘을 수 있음

def play_tts_wm8960(wav_or_mp3_path: str, card_id: Optional[int] = None) -> bool:
    """TTS 파일을 WM8960으로 재생. MP3면 mpg123/ffplay(기본 ALSA), WAV면 aplay -D 사용."""
    if not os.path.exists(wav_or_mp3_path):
        return False
    ext = os.path.splitext(wav_or_mp3_path)[1].lower()
    if ext == ".mp3":
        # 기본 ALSA 디바이스 사용 (WM8960을 default로 두면 됨)
        try:
            subprocess.run(["mpg123", "-q", wav_or_mp3_path], check=True, capture_output=True, timeout=TTS_PLAY_TIMEOUT)
            return True
        except subprocess.TimeoutExpired:
            log("⚠️ TTS 재생 시간 초과 (mpg123). 응답이 너무 길 수 있음.")
            return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_or_mp3_path],
                    check=True, capture_output=True, timeout=TTS_PLAY_TIMEOUT
                )
                return True
            except subprocess.TimeoutExpired:
                log("⚠️ TTS 재생 시간 초과 (ffplay).")
                return False
            except (subprocess.CalledProcessError, FileNotFoundError):
                log("⚠️ mpg123/ffplay 없음, espeak로 대체...")
                with open(wav_or_mp3_path, "rb") as f:
                    pass  # espeak는 텍스트 필요
                return False
    if ext == ".wav" and card_id is not None:
        devs = [f"plughw:{card_id},0", f"hw:{card_id},0", "default"]
        for dev in devs:
            r = subprocess.run(
                ["aplay", "-D", dev, "-q", wav_or_mp3_path],
                capture_output=True, timeout=TTS_PLAY_TIMEOUT
            )
            if r.returncode == 0:
                return True
    return False


def _play_keyword_ack_and_start_listening() -> None:
    """키워드 인식 시 '어, 인식했어' TTS 재생 후 LISTENING으로 전환 (별도 스레드에서 실행)"""
    try:
        ack_path = os.path.join(PARENT_DIR, "keyword_ack.mp3")
        client.audio.speech.create(
            model="tts-1", voice="alloy", input=KEYWORD_ACK_PHRASE
        ).stream_to_file(ack_path)
        play_tts_wm8960(ack_path, card_id=WM8960_CARD_ID)
    except Exception as e:
        log(f"⚠️ 키워드 인식 음성 재생 오류: {e}")
    finally:
        shared_state["status"] = "LISTENING"
        shared_state["text"] = "네, 말씀하세요!"
        shared_state["vad_speech_seen"] = False
        shared_state["vad_silence_frames"] = 0
        shared_state["vad_last_checked"] = 0


# ==================== AI (porcupine.py 동일) ====================
def web_search(query: str) -> str:
    log(f"🔎 [Tavily] 검색: {query}")
    try:
        if not tavily or len(TAVILY_API_KEY) < 10:
            return json.dumps({"error": "Tavily API 키가 설정되지 않았습니다"})
        results = tavily.search(query=query, search_depth="basic")
        if not results or "results" not in results:
            return json.dumps({"error": "검색 결과 없음"})
        context = [{"url": r["url"], "content": r["content"]} for r in results["results"]]
        return json.dumps(context)
    except Exception as e:
        log(f"❌ [Tavily] 오류: {e}")
        return json.dumps({"error": str(e)})

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "웹 검색",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }
]

WM8960_CARD_ID: Optional[int] = None  # audio_loop에서 설정 후 ai_worker에서 사용

def ai_worker(filename: str) -> None:
    global WM8960_CARD_ID
    try:
        log("📝 [1/4] STT 시작...")
        shared_state["text"] = "듣고 있어요..."
        with open(filename, "rb") as f:
            txt = client.audio.transcriptions.create(model="whisper-1", file=f, language="ko").text
        log(f"🗣️ [1/4] STT 완료 - User: {txt}")

        if not txt or not txt.strip():
            shared_state["text"] = "음성을 인식하지 못했습니다."
            return

        log("🤔 [2/4] LLM 처리 시작...")
        shared_state["text"] = f"🤔 {txt}"
        shared_state["status"] = "THINKING"
        msgs = [
            {"role": "system", "content": "너는 포큐파인(Porcupine)이야. 친구처럼 반말하고 짧게 대답해."},
            {"role": "user", "content": txt},
        ]
        res = client.chat.completions.create(model="gpt-4o", messages=msgs, tools=tools)
        msg = res.choices[0].message

        if msg.tool_calls:
            shared_state["text"] = "🌐 검색 중..."
            # assistant 메시지는 한 번만 추가 (tool_calls 포함)
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            }
            msgs.append(assistant_msg)
            # 각 tool_call_id에 대해 tool 메시지를 순서대로 추가 (전부 있어야 API 오류 없음)
            for tc in msg.tool_calls:
                if getattr(tc.function, "name", None) == "web_search":
                    arg = json.loads(tc.function.arguments or "{}")
                    search_res = web_search(arg.get("query", ""))
                else:
                    search_res = json.dumps({"error": "unknown tool"})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": search_res})
            final = client.chat.completions.create(model="gpt-4o", messages=msgs)
            reply = final.choices[0].message.content or ""
        else:
            reply = msg.content

        log(f"🤖 [2/4] LLM 완료 - Porcupine: {reply}")
        shared_state["text"] = reply
        shared_state["status"] = "SPEAKING"

        log("🔊 [3/4] TTS 생성...")
        tts_file = os.path.join(PARENT_DIR, "tts_output.mp3")
        client.audio.speech.create(model="tts-1", voice="alloy", input=reply).stream_to_file(tts_file)
        log("🔊 [4/4] 음성 재생 중 (WM8960)...")
        play_tts_wm8960(tts_file, card_id=WM8960_CARD_ID)
        log("✅ [4/4] TTS 완료.")
    except Exception as e:
        log(f"❌ AI Error: {e}")
        shared_state["text"] = f"오류: {str(e)[:50]}"
    finally:
        shared_state["status"] = "IDLE"
        shared_state["text"] = "Porcupine 대기 중..."
        shared_state["cooldown_until"] = time.time() + COOLDOWN_SEC_AFTER_RESPONSE

# ==================== 오디오 루프 (WM8960 + Porcupine) ====================
def audio_loop() -> None:
    global WM8960_CARD_ID
    if PICOVOICE_KEY == "..." or len(PICOVOICE_KEY) < 20:
        log("❌ Picovoice API 키가 설정되지 않았습니다.")
        shared_state["text"] = "Picovoice API 키를 설정해주세요"
        return

    WM8960_CARD_ID = get_wm8960_card()
    if WM8960_CARD_ID is None:
        log("❌ WM8960 카드를 찾을 수 없습니다. /boot/config.txt 에 dtoverlay=wm8960-soundcard 확인.")
        shared_state["text"] = "WM8960을 찾을 수 없습니다"
        return
    setup_wm8960_mixer(WM8960_CARD_ID)
    time.sleep(0.5)

    mic_index = get_wm8960_pyaudio_index()
    if mic_index is None:
        log("⚠️ WM8960 PyAudio 입력을 찾지 못함. 기본 디바이스(0) 사용.")
        mic_index = 0

    try:
        porcupine = pvporcupine.create(
            access_key=PICOVOICE_KEY,
            keywords=["porcupine"],
            sensitivities=[PORCUPINE_SENSITIVITY],
        )
    except Exception as e:
        log(f"❌ Picovoice 초기화 실패: {e}")
        shared_state["text"] = f"Picovoice 오류: {str(e)[:40]}"
        return

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
            input_device_index=mic_index,
        )
    except Exception as e:
        log(f"❌ 마이크 열기 실패: {e}")
        shared_state["text"] = f"마이크 오류: {str(e)[:50]}"
        porcupine.delete()
        pa.terminate()
        return

    shared_state["text"] = "Porcupine 대기 중..."
    frames: List[bytes] = []
    log("🎤 키워드 감지 대기 중... ('Porcupine' 말하거나 터미널에서 Enter 누르면 녹음)")

    while shared_state["is_running"]:
        try:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            # Porcupine에는 원본만 전달 (test_picovoice와 동일). 증폭은 녹음 저장할 때만.
            audio_raw = np.frombuffer(pcm, dtype=np.int16)
            shared_state["audio_chunk"] = audio_raw

            if shared_state["status"] == "IDLE":
                # 쿨다운 중이면 Porcupine 무시 (TTS 에코로 오인식 방지)
                if time.time() < shared_state.get("cooldown_until", 0):
                    pass
                else:
                    result = porcupine.process(tuple(audio_raw))
                    manual = shared_state.pop("manual_trigger", False)
                    if result >= 0 or manual:
                        if manual:
                            log("⌨️ 수동 트리거: 녹음 시작 (Enter)")
                        else:
                            log("✨ 'Porcupine' 키워드 감지!")
                        # 버퍼 비우기 (이전 데이터가 다음 질문에 섞이는 것 방지)
                        flush_frames = int(porcupine.sample_rate / porcupine.frame_length * FLUSH_SEC_AFTER_TRIGGER)
                        for _ in range(flush_frames):
                            try:
                                stream.read(porcupine.frame_length, exception_on_overflow=False)
                            except Exception:
                                break
                        shared_state["status"] = "ACK_PENDING"
                        shared_state["text"] = KEYWORD_ACK_PHRASE
                        frames = []
                        threading.Thread(target=_play_keyword_ack_and_start_listening, daemon=True).start()

            elif shared_state["status"] == "ACK_PENDING":
                # '어, 인식했어' 재생 중. 스트림만 소비하고, 재생 끝나면 스레드가 LISTENING으로 바꿈
                pass

            elif shared_state["status"] == "LISTENING":
                # test_record.py와 동일: 원본만 저장 (증폭 없음)
                frames.append(pcm)
                raw = np.frombuffer(b"".join(frames), dtype=np.int16)
                total_samples = len(raw)
                total_sec = total_samples / porcupine.sample_rate
                should_stop = False

                # 1) 최대 15초 도달 시 중지
                if total_sec >= MAX_RECORD_SEC:
                    should_stop = True
                    log("✅ 녹음 최대 길이(15초) 도달 -> AI 처리")
                else:
                    # 2) VAD: 구간별 에너지로 침묵 3초 이상이면 중지
                    samples_per_vad_chunk = int(porcupine.sample_rate * VAD_CHUNK_SEC)
                    silence_chunks_required = int(VAD_SILENCE_SEC / VAD_CHUNK_SEC)
                    last_checked = shared_state.get("vad_last_checked", 0)
                    while last_checked + samples_per_vad_chunk <= total_samples:
                        chunk = raw[last_checked : last_checked + samples_per_vad_chunk]
                        energy = np.abs(chunk).mean()
                        if energy >= VAD_THRESHOLD:
                            shared_state["vad_speech_seen"] = True
                            shared_state["vad_silence_frames"] = 0
                        else:
                            shared_state["vad_silence_frames"] = shared_state.get("vad_silence_frames", 0) + 1
                        last_checked += samples_per_vad_chunk
                    shared_state["vad_last_checked"] = last_checked

                    if shared_state.get("vad_speech_seen") and shared_state.get("vad_silence_frames", 0) >= silence_chunks_required:
                        should_stop = True
                        log("✅ 침묵 3초 이상 -> 녹음 중지, AI 처리")

                if should_stop:
                    raw = np.frombuffer(b"".join(frames), dtype=np.int16)
                    level = np.abs(raw).mean()
                    samples_per_chunk = int(porcupine.sample_rate * CHUNK_SEC)
                    n_chunks = max(1, len(raw) // samples_per_chunk)
                    chunks_above = 0
                    for i in range(n_chunks):
                        start = i * samples_per_chunk
                        end = min(start + samples_per_chunk, len(raw))
                        if end > start and np.abs(raw[start:end]).mean() >= CHUNK_LEVEL_THRESHOLD:
                            chunks_above += 1
                    if level < MIN_RECORD_LEVEL or chunks_above < MIN_CHUNKS_ABOVE_LEVEL:
                        log(f"⚠️ 녹음 레벨/말 구간 부족 (평균={level:.0f}, 말구간={chunks_above}/{n_chunks}), 말씀 없음으로 처리")
                        shared_state["status"] = "IDLE"
                        shared_state["text"] = "말씀이 없었습니다."
                        shared_state["cooldown_until"] = time.time() + COOLDOWN_SEC_AFTER_RESPONSE
                        frames = []
                    else:
                        log("✅ 녹음 완료 -> AI 처리 시작")
                        wav_path = os.path.join(PARENT_DIR, "input.wav")
                        with wave.open(wav_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(porcupine.sample_rate)
                            wf.writeframes(b"".join(frames))
                        threading.Thread(target=ai_worker, args=(wav_path,), daemon=True).start()
                        shared_state["status"] = "THINKING"
                        shared_state["text"] = "처리 중..."
                    frames = []
        except Exception:
            pass

    stream.stop_stream()
    stream.close()
    porcupine.delete()
    pa.terminate()

# ==================== LCD (wm8960_lcd_integrated_test / lcd_test 참조) ====================
LCD_WIDTH, LCD_HEIGHT = 240, 320
DC_PIN = 17
RESET_PIN = 4
BL_PIN = 25  # 백라이트 (GPIO18은 WM8960 I2S와 충돌 가능)
SPI_PORT, SPI_CS = 0, 0

ST7789_CASET = 0x2A
ST7789_RASET = 0x2B
ST7789_RAMWR = 0x2C

def lcd_send_command(spi, dc_pin: int, cmd: int) -> None:
    if LCD_AVAILABLE:
        GPIO.output(dc_pin, GPIO.LOW)
        spi.xfer([cmd])

def lcd_send_data(spi, dc_pin: int, data) -> None:
    if LCD_AVAILABLE:
        GPIO.output(dc_pin, GPIO.HIGH)
        spi.xfer([data] if isinstance(data, int) else list(data))

def lcd_init_st7789v(spi, dc_pin: int, rst_pin: int) -> None:
    if not LCD_AVAILABLE:
        return
    GPIO.output(rst_pin, GPIO.HIGH)
    time.sleep(0.001)
    GPIO.output(rst_pin, GPIO.LOW)
    time.sleep(0.010)
    GPIO.output(rst_pin, GPIO.HIGH)
    time.sleep(0.120)
    lcd_send_command(spi, dc_pin, 0x01)
    time.sleep(0.150)
    lcd_send_command(spi, dc_pin, 0x36)
    lcd_send_data(spi, dc_pin, 0x00)
    lcd_send_command(spi, dc_pin, 0xB2)
    lcd_send_data(spi, dc_pin, [0x0C, 0x0C, 0x00, 0x33, 0x33])
    lcd_send_command(spi, dc_pin, 0x3A)
    lcd_send_data(spi, dc_pin, 0x05)
    lcd_send_command(spi, dc_pin, 0xB7)
    lcd_send_data(spi, dc_pin, 0x14)
    lcd_send_command(spi, dc_pin, 0xBB)
    lcd_send_data(spi, dc_pin, 0x37)
    lcd_send_command(spi, dc_pin, 0xC0)
    lcd_send_data(spi, dc_pin, 0x2C)
    lcd_send_command(spi, dc_pin, 0xC2)
    lcd_send_data(spi, dc_pin, 0x01)
    lcd_send_command(spi, dc_pin, 0xC3)
    lcd_send_data(spi, dc_pin, 0x12)
    lcd_send_command(spi, dc_pin, 0xC4)
    lcd_send_data(spi, dc_pin, 0x20)
    lcd_send_command(spi, dc_pin, 0xD0)
    lcd_send_data(spi, dc_pin, [0xA4, 0xA1])
    lcd_send_command(spi, dc_pin, 0xC6)
    lcd_send_data(spi, dc_pin, 0x0F)
    lcd_send_command(spi, dc_pin, 0xE0)
    lcd_send_data(spi, dc_pin, [0xD0, 0x04, 0x0D, 0x11, 0x13, 0x2B, 0x3F, 0x54, 0x4C, 0x18, 0x0D, 0x0B, 0x1F, 0x23])
    lcd_send_command(spi, dc_pin, 0xE1)
    lcd_send_data(spi, dc_pin, [0xD0, 0x04, 0x0C, 0x11, 0x13, 0x2C, 0x3F, 0x44, 0x51, 0x2F, 0x1F, 0x1F, 0x20, 0x23])
    lcd_send_command(spi, dc_pin, 0x21)
    lcd_send_command(spi, dc_pin, 0x11)
    time.sleep(0.120)
    lcd_send_command(spi, dc_pin, 0x29)
    time.sleep(0.100)

def lcd_set_window(spi, dc_pin: int, x0: int, y0: int, x1: int, y1: int) -> None:
    if not LCD_AVAILABLE:
        return
    lcd_send_command(spi, dc_pin, ST7789_CASET)
    lcd_send_data(spi, dc_pin, [(x0 >> 8) & 0xFF, x0 & 0xFF, (x1 >> 8) & 0xFF, x1 & 0xFF])
    lcd_send_command(spi, dc_pin, ST7789_RASET)
    lcd_send_data(spi, dc_pin, [(y0 >> 8) & 0xFF, y0 & 0xFF, (y1 >> 8) & 0xFF, y1 & 0xFF])
    lcd_send_command(spi, dc_pin, ST7789_RAMWR)

def rgb565(r: int, g: int, b: int) -> int:
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)

# 글자마다 다른 색 (RGB). 인덱스로 순환 사용
LCD_CHAR_COLORS = [
    (255, 100, 100), (100, 255, 255), (255, 255, 100), (150, 255, 150),
    (255, 200, 150), (200, 150, 255), (100, 255, 200), (255, 150, 200),
    (200, 255, 100), (100, 200, 255), (255, 180, 100), (180, 255, 180),
]

def _char_width(draw, char: str, font) -> int:
    """한 글자 너비 픽셀"""
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        return max(bbox[2] - bbox[0], 1)
    except Exception:
        return 10

def draw_text_each_char_color(draw, x: int, y: int, s: str, font, color_index: int = 0) -> int:
    """문자열 s를 글자마다 다른 색으로 그리며 x 진행. 반환: 그린 뒤 x 위치."""
    for i, c in enumerate(s):
        color = LCD_CHAR_COLORS[(color_index + i) % len(LCD_CHAR_COLORS)]
        draw.text((x, y), c, fill=color, font=font)
        x += _char_width(draw, c, font)
    return x

def lcd_draw_image(spi, dc_pin: int, img: Image.Image) -> None:
    if not LCD_AVAILABLE:
        return
    if img.size != (LCD_WIDTH, LCD_HEIGHT):
        img = img.resize((LCD_WIDTH, LCD_HEIGHT), Image.LANCZOS)
    # LCD가 180도 뒤집혀 장착된 경우 회전 (화면이 정상 방향으로 보이도록)
    img = img.transpose(Image.ROTATE_180)
    pixel_data = []
    for y in range(LCD_HEIGHT):
        for x in range(LCD_WIDTH):
            r, g, b = img.getpixel((x, y))[:3]
            p = rgb565(r, g, b)
            pixel_data.extend([(p >> 8) & 0xFF, p & 0xFF])
    lcd_set_window(spi, dc_pin, 0, 0, LCD_WIDTH - 1, LCD_HEIGHT - 1)
    GPIO.output(dc_pin, GPIO.HIGH)
    for i in range(0, len(pixel_data), 4096):
        spi.xfer(pixel_data[i : i + 4096])

def lcd_show_title(spi, dc_pin: int) -> None:
    """LCD에 'PORCUPINE' / 'PROJECT' 두 줄로 고정 표시 (초기화 직후용)"""
    if not LCD_AVAILABLE or spi is None:
        return
    img = Image.new("RGB", (LCD_WIDTH, LCD_HEIGHT), color=(10, 10, 20))
    draw = ImageDraw.Draw(img)
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font_large = ImageFont.load_default()
    # 화면 중앙에 두 줄로 표시 (PROJECT에서 줄바꿈), 글자마다 다른 색
    line_height = 36
    block_h = line_height * 2
    y0 = (LCD_HEIGHT - block_h) // 2
    draw_text_each_char_color(draw, 15, y0, "PORCUPINE", font_large, 0)
    draw_text_each_char_color(draw, 15, y0 + line_height, "PROJECT", font_large, 9)
    lcd_draw_image(spi, dc_pin, img)


def lcd_update(spi, dc_pin: int) -> None:
    """shared_state 기준으로 LCD 화면 그리기"""
    if not LCD_AVAILABLE or spi is None:
        return
    status = shared_state["status"]
    text = shared_state["text"]
    mode_map = {"IDLE": "idle", "LISTENING": "recording", "THINKING": "thinking", "SPEAKING": "playing"}
    mode = mode_map.get(status, "idle")

    img = Image.new("RGB", (LCD_WIDTH, LCD_HEIGHT), color=(10, 10, 20))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_title = font_medium = font_small = ImageFont.load_default()

    y = 5
    draw_text_each_char_color(draw, 10, y, "Porcupine", font_title, 0)
    y += 24
    draw.line([(5, y), (235, y)], fill=(80, 80, 120), width=1)
    y += 8

    draw_text_each_char_color(draw, 10, y, f"Status: {status}", font_medium, 2)
    y += 20
    display_text = (text[:28] + "..") if len(text) > 28 else text
    draw_text_each_char_color(draw, 10, y, display_text, font_small, 5)

    draw.line([(5, 300), (235, 300)], fill=(80, 80, 120), width=1)
    draw_text_each_char_color(draw, 10, 305, "Porcupine say...", font_small, 7)

    lcd_draw_image(spi, dc_pin, img)

def _check_manual_trigger(timeout: float = 0.25) -> None:
    """stdin에서 Enter 입력 시 수동 트리거 설정 (Porcupine 대신 녹음 시작)"""
    try:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r and sys.stdin in r:
            sys.stdin.readline()
            shared_state["manual_trigger"] = True
            log("⌨️ 수동 트리거: 녹음 시작 (Enter)")
    except Exception:
        time.sleep(timeout)


def _ui_loop(lcd_spi, dc_pin: int, interval: float) -> None:
    """LCD 갱신 + Enter 키 확인 (백그라운드 스레드용)"""
    while shared_state["is_running"]:
        if lcd_spi is not None:
            try:
                lcd_update(lcd_spi, dc_pin)
            except Exception as e:
                log(f"LCD 업데이트 오류: {e}")
        _check_manual_trigger(interval)

# ==================== 메인 ====================
def main() -> None:
    # LCD 끄기: test_picovoice는 되는데 main만 안 될 때 LCD/SPI 간섭 확인용
    # (LCD 갱신 0.25초마다 = 이미지 생성·SPI 전송이 오디오 스레드와 CPU/버스 경쟁 가능)
    no_lcd = "--no-lcd" in sys.argv or os.environ.get("PORCUPINE_NO_LCD", "").lower() in ("1", "true", "yes")

    print("=" * 60)
    print(" Porcupine 챗봇 (WM8960 + LCD)" if not no_lcd else " Porcupine 챗봇 (WM8960, LCD 없음)")
    print("=" * 60)
    if no_lcd:
        print("\n[모드] --no-lcd: LCD 미사용 (간섭 확인용)\n")

    lcd_spi = None
    if LCD_AVAILABLE and not no_lcd:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(DC_PIN, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(RESET_PIN, GPIO.OUT, initial=GPIO.HIGH)
            GPIO.setup(BL_PIN, GPIO.OUT, initial=GPIO.HIGH)  # 백라이트 ON
            spi = spidev.SpiDev()
            spi.open(SPI_PORT, SPI_CS)
            spi.mode = 0
            spi.max_speed_hz = 500000
            lcd_init_st7789v(spi, DC_PIN, RESET_PIN)
            lcd_show_title(spi, DC_PIN)
            lcd_spi = spi
            log("LCD 초기화 완료.")
        except Exception as e:
            log(f"LCD 초기화 실패 (계속 진행): {e}")
            lcd_spi = None

    # LCD 스레드 미실행 (멀티스레드 시 Porcupine 인식 저하 방지)
    if not no_lcd:
        print("\n[안내] Porcupine이라고 말하면 녹음 시작됩니다. (LCD 갱신 스레드 비활성화)\n")
    else:
        print("\n[안내] Porcupine 말하거나 Enter로 녹음.\n")

    try:
        audio_loop()
    except KeyboardInterrupt:
        pass
    finally:
        shared_state["is_running"] = False
        time.sleep(0.5)
        if lcd_spi is not None and LCD_AVAILABLE:
            try:
                lcd_spi.close()
                GPIO.cleanup()
            except Exception:
                pass
        print("\n종료되었습니다.")

if __name__ == "__main__":
    main()
