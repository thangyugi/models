@echo off
REM Chạy backend + frontend trên Windows. Mở http://localhost:5173
start "Z-Image backend" cmd /k "cd backend && uvicorn server:app --host 127.0.0.1 --port 8000"
cd frontend
if not exist node_modules call npm install
npm run dev
