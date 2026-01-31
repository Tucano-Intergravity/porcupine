# Porcupine Voice Chatbot

Voice assistant using **Picovoice Porcupine** (wake word), **OpenAI Whisper** (STT), **GPT-4o** (LLM), and **OpenAI TTS**, with **WM8960** audio HAT and optional **ST7789V LCD** on Raspberry Pi.

## Features

- Wake word: say "Porcupine" (or use manual trigger) to start recording
- Short voice acknowledgment ("어 말해봐") before recording
- 3–4 second recording (configurable), then STT → LLM → TTS
- WM8960 mixer setup and raw 16 kHz mono recording (same as test_record)
- Optional LCD status display (can be disabled with `--no-lcd` to avoid thread interference)
- Noise gate: skip STT when recording level is too low or no speech segments
- Korean STT (`language="ko"`) to avoid wrong-language transcription

## Requirements

- Raspberry Pi with WM8960 Audio HAT (and optionally ST7789V LCD)
- Python 3 + `openai`, `tavily-python`, `pvporcupine`, `pyaudio`, `numpy`, `Pillow`
- API keys in parent folder: `OpenAI_API_Key.txt`, `tavily_API_key.txt`, `picovoice_API_Key.txt`

## Usage

```bash
cd porcupine
python3 main.py
```

- Run without LCD: `python3 main.py --no-lcd`
- Put API key files in the parent directory (`pi_ai/`).

## Scripts

- `main.py` – main chatbot (Porcupine + WM8960 + optional LCD)
- `test_record.py` – record 5 s and play back (same 16 kHz mono as main)
- `test_picovoice.py` – check Picovoice API and optional mic keyword test

## Config (top of main.py)

- `KEYWORD_ACK_PHRASE` – voice reply after wake word (e.g. "어 말해봐")
- `RECORD_SEC` – recording length in seconds
- `MIN_RECORD_LEVEL`, `CHUNK_LEVEL_THRESHOLD`, `MIN_CHUNKS_ABOVE_LEVEL` – noise gate
- `FLUSH_SEC_AFTER_TRIGGER`, `COOLDOWN_SEC_AFTER_RESPONSE` – timing
