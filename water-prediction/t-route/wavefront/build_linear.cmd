@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
nvcc -O3 -arch=sm_86 -std=c++17 --expt-relaxed-constexpr linear_mc.cu -o linear_mc.exe
exit /b %errorlevel%
