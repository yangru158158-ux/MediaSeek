@echo off
rem Local logic assertions (Windows): 55 query-understanding checks + Python mirror.
rem Runs in seconds before pushing - no macOS CI queue (created 2026-10-23 after
rem repeated capacity rejections on macOS runners during US peak hours).
rem One-time deps:
rem   winget install Swift.Toolchain          (installs to %%LocalAppData%%\Programs\Swift)
rem   VS Build Tools 2022 + C++ workload      (swiftc linking needs MSVC env)
rem Usage: cmd /c scripts\local_logic_tests.cmd
setlocal
set "SWROOT=%LocalAppData%\Programs\Swift"
set "TC="
for /d %%D in ("%SWROOT%\Toolchains\*") do if not defined TC set "TC=%%D"
set "PLAT="
for /d %%D in ("%SWROOT%\Platforms\*") do if not defined PLAT set "PLAT=%%D"
set "RT="
for /d %%D in ("%SWROOT%\Runtimes\*") do if not defined RT set "RT=%%D"
if not defined TC echo [SKIP] Swift toolchain not found. Run: winget install Swift.Toolchain & exit /b 1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%TC%\usr\bin;%RT%\usr\bin;%PATH%"
set "SDKROOT=%PLAT%\Windows.platform\Developer\SDKs\Windows.sdk"
cd /d "%~dp0.."

echo == Query understanding 55 assertions (local swiftc) ==
swiftc -O -o "%TEMP%\mediaseek_qtest.exe" scripts\query_understanding_test.swift MediaSeek\Core\QueryUnderstanding.swift MediaSeek\Core\LabelZh.swift MediaSeek\Core\LabelZhExtra.swift
if errorlevel 1 echo [FAIL] compile error & exit /b 1
"%TEMP%\mediaseek_qtest.exe"
if errorlevel 1 exit /b 1

echo == Python mirror exhaustive (dictionary/labels/generic) ==
python scripts\label_tokens_mirror.py
