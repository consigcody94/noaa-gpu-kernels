@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -O3 -arch=sm_120 -o "%~dp0ccpp_tridi1_realdata.exe" "%~dp0ccpp_tridi1_realdata.cu"
if errorlevel 1 (echo BUILD FAIL) else (echo BUILD OK)
