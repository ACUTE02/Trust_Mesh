@echo off
rem TrustMesh: start everything with one double-click.
rem Starts Docker Desktop if needed, brings the stack up, waits for the models, opens the site.
cd /d "%~dp0"

docker info >nul 2>&1
if errorlevel 1 (
    echo Starting Docker Desktop...
    docker desktop start >nul 2>&1
    if errorlevel 1 start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
)

echo Waiting for Docker...
set /a tries=0
:wait_docker
docker info >nul 2>&1
if not errorlevel 1 goto docker_ready
set /a tries+=1
if %tries% geq 60 (
    echo Docker did not start. Open Docker Desktop by hand, wait until it says "running", then run this file again.
    pause
    exit /b 1
)
timeout /t 3 /nobreak >nul
goto wait_docker

:docker_ready
echo Starting TrustMesh (the first run builds the images and can take several minutes)...
docker compose up -d
if errorlevel 1 (
    echo docker compose failed. See the message above.
    pause
    exit /b 1
)

echo Waiting for the models to load (usually under a minute)...
set /a tries=0
:wait_api
curl -s -m 5 http://localhost:8080/api/health | findstr /c:"\"voice\":{\"ready\":true" >nul 2>&1
if not errorlevel 1 goto ready
set /a tries+=1
if %tries% geq 60 (
    echo The site is up but the models are still loading or failed. Check: docker compose logs api --tail 30
    goto open
)
timeout /t 5 /nobreak >nul
goto wait_api

:ready
echo Ready.
:open
start "" http://localhost:8080
echo.
echo Site:   http://localhost:8080
echo To stop: double-click stop.bat  (or run: docker compose down)
pause
