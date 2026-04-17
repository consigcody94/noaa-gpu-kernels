@echo off
REM Build script: calls vcvars64 then nvcc to compile wavefront_mc.cu
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo vcvars64 failed
    exit /b 1
)
nvcc -O3 -arch=sm_86 -std=c++17 --expt-relaxed-constexpr wavefront_mc.cu -o wavefront_mc.exe
exit /b %errorlevel%
