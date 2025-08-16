# GPU consume_point_schedule library with C++20 support
NVCC = nvcc
CXX = g++

# C++20 support for both host and device code
NVCC_FLAGS = -std=c++20 -O3 -arch=sm_61 --compiler-options '-std=c++20 -O3 -fPIC' -shared
CUDA_LIBS = -lcudart

# Target libraries  
CONSUME_TARGET = libgpu_consume_point_schedule.so
CONSUME_SOURCE = gpu_consume_point_schedule.cu

# Default target
all: $(CONSUME_TARGET)

# Build the GPU consume_point_schedule library
$(CONSUME_TARGET): $(CONSUME_SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(CONSUME_TARGET) $(CONSUME_SOURCE) $(CUDA_LIBS)

# Clean build artifacts
clean:
	rm -f $(CONSUME_TARGET)

# Install to common locations for testing
install: $(CONSUME_TARGET)
	cp $(CONSUME_TARGET) .
	cp $(CONSUME_TARGET) /home/puckapao/ 2>/dev/null || true
	cp $(CONSUME_TARGET) ../../ 2>/dev/null || true

.PHONY: all clean install
