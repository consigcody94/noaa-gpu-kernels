@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe
cd /d U:\AI\_noaa-research\noaa-gpu-kernels\_realdata\owp-hydrology
"%NVCC%" -O3 -arch=sm_120 -o owp_batched_realdata.exe owp_batched_kernels_realdata.cu
if errorlevel 1 exit /b 1
"%NVCC%" -O3 -arch=sm_120 -o owp_extended_realdata.exe owp_extended_kernels_realdata.cu
if errorlevel 1 exit /b 1
"%NVCC%" -O3 -arch=sm_120 -o owp_snow17_lgar_realdata.exe owp_snow17_lgar_realdata.cu
if errorlevel 1 exit /b 1
echo BUILD OK
