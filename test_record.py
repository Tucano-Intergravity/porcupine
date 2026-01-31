#!/usr/bin/env python3
"""
WM8960 마이크 녹음 테스트

챗봇과 동일한 설정(16kHz, 모노)으로 몇 초 녹음 후 파일로 저장·재생해서
녹음이 제대로 되는지 확인하는 스크립트입니다.
"""

import os
import sys
import time
import wave
import subprocess
import numpy as np
import pyaudio

# 상위 폴더(pi_ai) 기준
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(PARENT_DIR)

# Porcupine과 동일: 16kHz 모노
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SEC = 5
OUTPUT_WAV = os.path.join(PARENT_DIR, "test_record.wav")


def get_wm8960_card():
    try:
        r = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
        for line in r.stdout.split("\n"):
            if "wm8960" in line.lower() and "card" in line.lower():
                return int(line.split("card")[1].split(":")[0].strip())
    except Exception:
        pass
    return None


def amixer_set(card_id, name, value):
    r = subprocess.run(
        ["amixer", "-c", str(card_id), "set", name, value],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return r.returncode == 0


def setup_wm8960_mixer(card_id):
    amixer_set(card_id, "Mono Output Mixer Left", "on")
    amixer_set(card_id, "Mono Output Mixer Right", "on")
    amixer_set(card_id, "Speaker", "80%")
    amixer_set(card_id, "Speaker Playback Volume", "80%")
    amixer_set(card_id, "Headphone", "100%")
    amixer_set(card_id, "Headphone Playback Volume", "100%")
    amixer_set(card_id, "Playback", "100%")
    amixer_set(card_id, "Playback Volume", "100%")
    amixer_set(card_id, "Left Boost Mixer LINPUT1", "on")
    amixer_set(card_id, "Right Boost Mixer RINPUT1", "on")
    amixer_set(card_id, "Capture", "100%")
    amixer_set(card_id, "Capture Volume", "100%")
    amixer_set(card_id, "ADC PCM Capture Volume", "100%")
    amixer_set(card_id, "Left Input Boost Mixer LINPUT1 Volume", "3")
    amixer_set(card_id, "Right Input Boost Mixer RINPUT1 Volume", "3")


def get_wm8960_pyaudio_index():
    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = (info.get("name") or "").lower()
            if info.get("maxInputChannels", 0) > 0 and "wm8960" in name:
                return i
    finally:
        pa.terminate()
    return None


def main():
    print("=" * 60)
    print(" WM8960 녹음 테스트 (16kHz 모노, 챗봇과 동일)")
    print("=" * 60)

    card_id = get_wm8960_card()
    if card_id is None:
        print("\n[오류] WM8960 카드를 찾을 수 없습니다.")
        print("  aplay -l 로 카드 확인, /boot/config.txt 에 dtoverlay=wm8960-soundcard 확인")
        sys.exit(1)
    print(f"\n[1] WM8960 카드: {card_id}")

    print("[2] WM8960 믹서 설정 중...")
    setup_wm8960_mixer(card_id)
    time.sleep(0.5)

    mic_index = get_wm8960_pyaudio_index()
    if mic_index is None:
        print("\n[경고] WM8960 PyAudio 입력을 찾지 못함. 디바이스 0 사용.")
        mic_index = 0
    else:
        print(f"[3] WM8960 마이크 PyAudio 인덱스: {mic_index}")

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            rate=SAMPLE_RATE,
            channels=CHANNELS,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=512,
            input_device_index=mic_index,
        )
    except Exception as e:
        print(f"\n[오류] 마이크 열기 실패: {e}")
        print("  - 16kHz 미지원일 수 있음. 이 카드가 지원하는 샘플레이트 확인: arecord -D hw:N,0 --dump-hw-params")
        pa.terminate()
        sys.exit(1)

    print(f"\n[4] {RECORD_SEC}초 녹음 중... (마이크에 대고 말해보세요)")
    frames = []
    for _ in range(0, int(SAMPLE_RATE / 512 * RECORD_SEC)):
        data = stream.read(512, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()

    with wave.open(OUTPUT_WAV, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    print(f"\n[5] 저장 완료: {OUTPUT_WAV}")

    # 레벨 확인 (대략적인 크기)
    raw = np.frombuffer(b"".join(frames), dtype=np.int16)
    level = np.abs(raw).mean()
    peak = np.abs(raw).max()
    print(f"     평균 레벨: {level:.0f} / 32768, 피크: {peak}")
    if level < 100:
        print("     ※ 레벨이 매우 낮음. 마이크 감도/거리 확인.")
    elif level > 20000:
        print("     ※ 레벨이 매우 높음 (클리핑 가능).")
    else:
        print("     → 레벨 정상 범위로 보임.")

    print("\n[6] 재생할까요? (WM8960 스피커로 재생)")
    try:
        dev = f"plughw:{card_id},0"
        r = subprocess.run(
            ["aplay", "-D", dev, "-q", OUTPUT_WAV],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            print("     재생 완료. 들리는 소리가 방금 녹음한 것이 맞나요?")
        else:
            print(f"     재생 실패 (코드 {r.returncode}). aplay -D plughw:{card_id},0 {OUTPUT_WAV} 로 직접 시도.")
    except FileNotFoundError:
        print("     aplay 없음. 파일만 확인하세요.")
    except Exception as e:
        print(f"     재생 오류: {e}")

    print("\n" + "=" * 60)
    print(" 테스트 끝. 녹음이 잘 들리면 챗봇 쪽은 같은 경로로 녹음 중입니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
