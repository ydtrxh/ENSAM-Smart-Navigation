@echo off
title ENSAM Navigation - Streamlit Server
cd /d "%~dp0"
echo ============================================
echo   ENSAM Navigation App - Demarrage...
echo ============================================
echo.
echo  Serveur disponible sur : http://localhost:8501
echo  Gardez cette fenetre ouverte.
echo  Fermez-la pour arreter le serveur.
echo.
python -m streamlit run app/main.py
pause
