#!/bin/zsh
# Boost: запуск сервера + Cloudflare-туннеля для доступа с телефона.
# Использование: ./start_remote.sh   (печатает ссылку с ключом)
cd "$(dirname "$0")"
export PATH="$PWD/.bin:$PATH"

KEY_FILE=".access_key"
[ -f "$KEY_FILE" ] || python3 -c "import secrets; print(secrets.token_urlsafe(12))" > "$KEY_FILE"
KEY=$(cat "$KEY_FILE")

if ! curl -s -m 2 -o /dev/null "http://127.0.0.1:8000/api/jobs?key=$KEY"; then
  pkill -f "uvicorn boost.web.app" 2>/dev/null
  sleep 1
  BOOST_ACCESS_KEY="$KEY" nohup caffeinate -is .venv/bin/python -m uvicorn boost.web.app:app \
    --host 127.0.0.1 --port 8000 > /tmp/boost_uvicorn.log 2>&1 &
  sleep 3
fi

pkill -f "cloudflared tunnel" 2>/dev/null
sleep 1
rm -f /tmp/cloudflared.log
nohup .bin/cloudflared tunnel --url http://127.0.0.1:8000 > /tmp/cloudflared.log 2>&1 &
for i in {1..30}; do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/cloudflared.log | head -1)
  [ -n "$URL" ] && break
  sleep 2
done

if [ -n "$URL" ]; then
  echo ""
  echo "  Ссылка для телефона (открой один раз — дальше пустит по куке):"
  echo "  $URL/?key=$KEY"
  echo ""
else
  echo "Туннель не поднялся — смотри /tmp/cloudflared.log"
fi
