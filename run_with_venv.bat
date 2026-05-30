@echo off
REM Activate .venv310 and run the application (Windows CMD)
if not exist ".venv310\Scripts\activate.bat" (
  echo Virtual environment .venv310 not found.
  exit /b 1
)
call .venv310\Scripts\activate.bat
python run_app.py