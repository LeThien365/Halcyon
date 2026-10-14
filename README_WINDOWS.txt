Halcyon 1.8.0 - Source for Windows
==================================

Engine UCI chess, C++17. Khong can thu vien ngoai (chi standard library).

CACH BUILD
----------

1) MinGW-w64 (khuyen nghi, de)
   - Cai MinGW-w64 hoac MSYS2
   - Mo terminal, vao thu muc halycon_src
   - Chay:  build_mingw.bat
   - Ket qua: halcyon.exe

2) Visual Studio (MSVC)
   - Cai Visual Studio (workload "Desktop development with C++")
   - Mo "x64 Native Tools Command Prompt for VS"
   - cd den thu muc halycon_src
   - Chay:  build_msvc.bat
   - Ket qua: halcyon.exe

3) Manual MinGW:
   g++ -O3 -std=c++17 -pthread -DNDEBUG -march=native -DHALCYON_HIDDEN_SIZE=256 -Iinclude src\*.cpp -o halcyon.exe

CHAY VOI GUI
------------
Dung Arena, Cute Chess, Banksia, Scid vs PC...
Them engine: chon file halcyon.exe, protocol UCI.

GHI CHU
-------
- HALCYON_HIDDEN_SIZE=256 bat buoc de load mang NNUE mac dinh nhung trong source.
- AVX2 tang toc NNUE; neu CPU cu, bo -DHALCYON_USE_AVX2 /arch:AVX2.
- Linux: dung Makefile (make).
