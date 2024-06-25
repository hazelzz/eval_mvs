@echo off
SET DISTUTILS_USE_SDK=1
SET MSSdk=1
SET "VS_VERSION=17.0"
SET "VS_MAJOR=17"
SET "VS_YEAR=2022"
SET "MSYS2_ARG_CONV_EXCL=/AI;/AL;/OUT;/out"
SET "MSYS2_ENV_CONV_EXCL=CL"
SET "PY_VCRUNTIME_REDIST=\bin\vcruntime140.dll"
SET "CXX=cl.exe"
SET "CC=cl.exe"
SET "VSINSTALLDIR=C:\Program Files\Microsoft Visual Studio\2022\Community\"
SET "NEWER_VS_WITH_OLDER_VC=0"
SET "WindowsSDKVer=10.0.20348.0"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
