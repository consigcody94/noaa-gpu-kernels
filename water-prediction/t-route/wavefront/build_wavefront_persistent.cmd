@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
nvcc -O3 -arch=sm_86 -std=c++17 -rdc=true --expt-relaxed-constexpr -Xcompiler "/Zc:preprocessor" wavefront_persistent.cu -o wavefront_persistent.exe
exit /b %errorlevel%
