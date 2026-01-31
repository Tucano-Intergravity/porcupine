#!/usr/bin/env python3
"""
Picovoice(Porcupine) 서비스 동작 확인

1) API 키 유효성 + 엔진 초기화
2) 선택: 마이크로 몇 초 녹음 후 키워드 감지 테스트
"""

import os
import sys
import time
import numpy as np
import pyaudio
import pvporcupine

# 상위 폴더(pi_ai)에서 API 키 로드
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(PARENT_DIR)

KEY_FILE = "picovoice_API_Key.txt"
KEY_PATH = os.path.join(PARENT_DIR, KEY_FILE)


def load_key():
    if not os.path.exists(KEY_PATH):
        return None
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        key = f.read().strip().replace("\n", "").replace("\r", "").replace(" ", "")
    return key if key and len(key) > 20 else None


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
    return 0


def main():
    print("=" * 60)
    print(" Picovoice(Porcupine) 서비스 확인")
    print("=" * 60)

    key = load_key()
    if not key:
        print(f"\n❌ API 키 없음 또는 짧음: {KEY_PATH}")
        print("   Picovoice 콘솔(https://console.picovoice.ai/)에서 키 발급 후 파일에 저장")
        sys.exit(1)
    print(f"\n[1] API 키 로드됨 (길이: {len(key)})")

    # --- 엔진 초기화 (키 유효성 + 서비스 연동 확인) ---
    print("\n[2] Porcupine 엔진 초기화 중...")
    try:
        porcupine = pvporcupine.create(
            access_key=key,
            keywords=["porcupine"],
            sensitivities=[1.0],
        )
    except Exception as e:
        print(f"❌ Porcupine 초기화 실패: {e}")
        print("   - API 키 만료/잘못됨, 네트워크 문제, 할당량 초과 등 가능")
        sys.exit(1)

    print("✅ Porcupine 초기화 성공 (Picovoice 서비스 정상)")
    print(f"   샘플레이트: {porcupine.sample_rate} Hz")
    print(f"   프레임 길이: {porcupine.frame_length} 샘플")

    # --- 선택: 마이크로 키워드 감지 테스트 ---
    print("\n[3] 마이크로 키워드 감지 테스트 (10초)? [y/N]: ", end="", flush=True)
    try:
        line = sys.stdin.readline().strip().lower()
    except Exception:
        line = "n"
    if line != "y" and line != "yes":
        porcupine.delete()
        print("\n테스트 건너뜀. 엔진 초기화까지 정상이면 Picovoice 서비스는 동작 중입니다.")
        print("=" * 60)
        return

    mic_index = get_wm8960_pyaudio_index()
    if mic_index != 0:
        print(f"   WM8960 마이크 사용 (인덱스 {mic_index})")
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
        print(f"❌ 마이크 열기 실패: {e}")
        porcupine.delete()
        pa.terminate()
        sys.exit(1)

    print("   10초 동안 'Porcupine'이라고 말해보세요...")
    start = time.time()
    detected = 0
    while time.time() - start < 10:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        audio = np.frombuffer(pcm, dtype=np.int16)
        result = porcupine.process(tuple(audio))
        if result >= 0:
            detected += 1
            print("   ✨ 키워드 감지!")

    stream.stop_stream()
    stream.close()
    porcupine.delete()
    pa.terminate()

    print(f"\n   결과: 10초 동안 {detected}회 감지")
    if detected == 0:
        print("   ※ 한 번도 안 잡혀도 엔진은 정상. 감도/마이크/발음 조정 후 다시 시도.")
    print("=" * 60)


if __name__ == "__main__":
    main()
