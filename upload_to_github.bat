@echo off
echo =========================================
echo  GitHub Auto Upload Script
echo =========================================
echo.

git add .
git commit -m "Auto update via upload_to_github.bat"
git pull --no-edit
git push

echo.
echo =========================================
echo  Upload completed! You can close this window.
echo =========================================
pause
