@echo off
echo.
echo =========================================
echo    🏆 Transfermarkt Analiz Dashboard    
echo =========================================
echo.
echo 🚀 Dashboard başlatılıyor...
echo 📡 Tarayıcınızda otomatik olarak açılacak
echo.

cd /d "%~dp0"

REM Python environment kontrolü
C:/Users/cengi/anaconda3/Scripts/conda.exe run -p C:\Users\cengi\anaconda --no-capture-output python -c "import streamlit; print('✅ Streamlit hazır!')"

if %ERRORLEVEL% neq 0 (
    echo ❌ Streamlit bulunamadı!
    echo 💡 Lütfen önce 'pip install streamlit' komutunu çalıştırın
    pause
    exit /b 1
)

echo.
echo 🌐 Dashboard açılıyor: http://localhost:8501
echo ⏹️  Kapatmak için Ctrl+C kullanın
echo.

REM Streamlit uygulamasını başlat
C:/Users/cengi/anaconda3/Scripts/conda.exe run -p C:\Users\cengi\anaconda --no-capture-output streamlit run main.py --server.port=8501 --server.address=localhost --browser.gatherUsageStats=false

pause
