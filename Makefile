# Compiler toolchains
CC       := g++
CC_C     := gcc
NVCC     := nvcc

# Flags
CXXFLAGS := -O3 -std=c++11
CFLAGS   := -O3 -Wall -Wextra -pthread -mrdseed -std=c11
NVCCFLAGS:= -O3 -std=c++11

# Target
TARGET   := anneal

# Object files
OBJS     := main.o annealing.o annealing_cuda.o

# Default rule
all: $(TARGET)

# Compile the C annealing host code
annealing.o: annealing.c annealing.h
	$(CC_C) $(CFLAGS) -c $< -o $@

# Compile the C++ driver
main.o: main.cpp annealing.h
	$(CC) $(CXXFLAGS) -c $< -o $@

# Compile the CUDA kernels
annealing_cuda.o: annealing.cu annealing.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link everything with nvcc (so CUDA runtime is linked correctly)
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
