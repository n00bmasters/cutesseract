
NVCC ?= nvcc
CXX ?= g++
LD = g++

CXX_FLAGS = -O2 -std=c++20 -Iinclude
BUILD_DIR := build
SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
TARGET := $(BUILD_DIR)/main



all: build $(TARGET)

build:
	mkdir -p build

# build/$(wildcard: *.cxx.o) : src/$(wildcard: *.cpp)
# 	$(CXX) -c $< $(CXX_FLAGS) -o $@

$(TARGET): $(OBJECTS)
	$(LD) $^ -o $@

$(BUILD_DIR)/%.o : src/%.cpp | build
	$(CXX) -c $< $(CXX_FLAGS) -o $@


format:
	clang-format -i $(SOURCES) treap.hpp

clean:
	rm -f $(TARGET) $(OBJECTS)
