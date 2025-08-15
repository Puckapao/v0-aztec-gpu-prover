# CUDA MSM library with C++20 support
NVCC = nvcc
CXX = g++

# C++20 support for both host and device code
NVCC_FLAGS = -std=c++20 -O3 -arch=sm_61 --compiler-options '-std=c++20 -O3 -fPIC' -shared
CUDA_LIBS = -lcudart

# Target library
TARGET = libcuda_msm.so
SOURCE = cuda_msm.cu

# Default target
all: $(TARGET)

# Build the shared library with C++20
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE) $(CUDA_LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Install to common locations for testing
install: $(TARGET)
	cp $(TARGET) .
	cp $(TARGET) /home/puckapao/ 2>/dev/null || true
	cp $(TARGET) ../../ 2>/dev/null || true

.PHONY: all clean install
