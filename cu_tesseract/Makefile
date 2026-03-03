
NVCC ?= nvcc
CXX ?= g++
LD = g++

CXX_FLAGS = -O2 -std=c++20



all: build/ build/main

build/:
	mkdir -p build

# build/$(wildcard: *.cxx.o) : src/$(wildcard: *.cpp)
# 	$(CXX) -c $< $(CXX_FLAGS) -o $@

build/main : build/main.o
	$(LD) $< -o $@

build/main.o : src/main.cpp
	$(CXX) -c $< $(CXX_FLAGS) -o $@


format:
	clang-format -i $(SOURCES) treap.hpp

clean:
	rm -f $(TARGET)
