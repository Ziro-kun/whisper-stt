@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0run.ps1"
if errorlevel 1 pause
