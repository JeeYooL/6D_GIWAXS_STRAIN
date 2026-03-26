@echo off
echo Starting GIWAX Strain Analyzer...
call .\strain\Scripts\activate.bat
streamlit run test.py
pause
