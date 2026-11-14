#!/usr/bin/env bash
# Chạy backend (port 8000) + frontend dev (port 5173). Mở http://localhost:5173
set -e
cd "$(dirname "$0")"
(cd backend && uvicorn server:app --host 127.0.0.1 --port 8000) &
BACK=$!
trap "kill $BACK 2>/dev/null" EXIT
cd frontend
[ -d node_modules ] || npm install
npm run dev
