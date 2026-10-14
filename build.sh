#!/bin/bash
# Linux/macOS:  ./build.sh [hidden_size]     (mặc định 1024)
# Bỏ -mavx2 -mfma nếu CPU không hỗ trợ AVX2 (engine vẫn đúng, chỉ chậm hơn).
set -e
H=${1:-1024}
g++ -std=c++17 -O2 -mavx2 -mfma -DHALCYON_HIDDEN_SIZE=$H -Iinclude src/*.cpp -o halcyon -lpthread
echo "Build xong: ./halcyon (hidden $H)"
