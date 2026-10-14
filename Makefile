# Halcyon 2.0 — portable by default (SSE2), optional native AVX2
CXX      ?= g++
HIDDEN   ?= 1024
SRC      := $(wildcard src/*.cpp)
INC      := -Iinclude

# Generic x86-64: no AVX-512, works on Windows MinGW and Linux
CXXFLAGS_PORTABLE := -O3 -std=c++17 -pthread -DNDEBUG -msse2 -m64 -DHALCYON_HIDDEN_SIZE=$(HIDDEN)
# Host CPU (may enable AVX2 if available)
CXXFLAGS_NATIVE   := -O3 -std=c++17 -pthread -DNDEBUG -march=native -DHALCYON_HIDDEN_SIZE=$(HIDDEN)

.PHONY: all portable native clean

all: portable

portable:
	$(CXX) $(CXXFLAGS_PORTABLE) $(INC) $(SRC) -o halcyon

native:
	$(CXX) $(CXXFLAGS_NATIVE) $(INC) $(SRC) -o halcyon

clean:
	rm -f halcyon halcyon.exe
