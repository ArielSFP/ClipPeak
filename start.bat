@echo off
REM %~dp0 = the folder this .bat file lives in

REM --- Window 1: run the website (npm run dev in the 'website' folder)
start "website" cmd /k "cd /d %~dp0website && npm run dev"

REM --- Window 2: run the server (uvicorn reload in the project root)
start "server" cmd /k "cd /d %~dp0 && python -m uvicorn api:app --reload"
