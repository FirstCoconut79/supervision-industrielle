@echo off
title Supervision industrielle
cd /d "%~dp0"
echo Demarrage de l application de supervision...
echo.
echo Une page va s ouvrir dans le navigateur.
echo Pour utiliser un autre appareil : reperer la ligne "Network URL" ci-dessous
echo (ou ouvrir le menu "Acces depuis un autre appareil" dans l application).
echo Pour arreter l application : Ctrl + C, ou fermer cette fenetre.
echo.
python -m streamlit run app.py
pause
