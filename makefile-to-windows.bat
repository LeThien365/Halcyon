@echo off
setlocal
echo Dang build Halcyon STATIC - chi x86-64 (khong AVX2)...

set PATH=%~dp0msys64\ucrt64\bin;%~dp0msys64\mingw64\bin;%PATH%

set CXX=g++
where g++ >nul 2>&1
if errorlevel 1 set CXX=D:\my_engine\halcyon\msys64\mingw64\bin\g++.exe

%CXX% -std=c++17 -O3 -DNDEBUG ^
    -march=x86-64 -msse2 ^
    -DHALCYON_HIDDEN_SIZE=1024 ^
    -Iinclude ^
    src/attacks.cpp ^
    src/bitboard.cpp ^
    src/eval.cpp ^
    src/main.cpp ^
    src/movegen.cpp ^
    src/nnue.cpp ^
    src/polyglot.cpp ^
    src/position.cpp ^
    src/search.cpp ^
    src/see.cpp ^
    src/zobrist.cpp ^
    src/tablebase.cpp ^
    -o haliyon_x64.exe ^
    -static -pthread -static-libgcc -static-libstdc++

if exist haliyon_x64.exe (
    echo [OK] Build xong: haliyon_x64.exe
) else (
    echo [LOI] Build fail.
)
pause