@echo off
rem TrustMesh: stop the stack. Data (the Neo4j graph, downloaded models) is kept for the next start.
cd /d "%~dp0"
docker compose down
pause
