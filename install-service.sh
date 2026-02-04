#!/bin/bash
# 부팅 시 porcupine main.py 자동 실행 서비스 설치
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="porcupine.service"

echo "서비스 파일 복사 중..."
sudo cp "$SCRIPT_DIR/porcupine.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "설치 완료. 부팅 시 자동 실행됩니다."
echo "  지금 시작: sudo systemctl start $SERVICE_NAME"
echo "  상태 확인: sudo systemctl status $SERVICE_NAME"
echo "  로그 확인: journalctl -u $SERVICE_NAME -f"
