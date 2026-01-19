@echo off
title Building LocalImageSearch Tool...
echo ===================================================
echo     BUILDING WINDOWS EXECUTABLE
echo ===================================================

echo [1/3] Installing PyInstaller...
pip install pyinstaller

echo [2/3] Building application (This may take a few minutes)...
python -m PyInstaller build_app.spec --clean --noconfirm --distpath "D:/"

echo.
echo ===================================================
echo     BUILD COMPLETE
echo ===================================================
echo.
echo Your app is located in: D:\LocalImageSearch
echo.
pause
