# CUDA MSM library with C++20 support
NVCC = nvcc
CXX = g++

# C++20 support for both host and device code
NVCC_FLAGS = -std=c++20 -O3 -arch=sm_61 --compiler-options '-std=c++20 -O3 -fPIC' -shared
CUDA_LIBS = -lcudart

# Target libraries  
TARGET = libcuda_msm.so
DEBUG_TARGET = libdebug_msm.so
PIPPENGER_TARGET = libcuda_msm_pippenger.so
HYBRID_TARGET = libcuda_msm_hybrid.so
DIRECT_TARGET = libcuda_msm_direct.so
ACCELERATION_TARGET = libgpu_affine_acceleration.so
SOURCE = cuda_msm.cu
DEBUG_SOURCE = debug_msm.cu
PIPPENGER_SOURCE = cuda_msm_pippenger.cu
HYBRID_SOURCE = cuda_msm_hybrid.cu
DIRECT_SOURCE = cuda_msm_direct.cu
ACCELERATION_SOURCE = gpu_affine_acceleration.cu

# Default target
all: $(TARGET) $(DEBUG_TARGET) $(PIPPENGER_TARGET) $(HYBRID_TARGET) $(DIRECT_TARGET) $(ACCELERATION_TARGET)

# Build the main CUDA MSM library
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE) $(CUDA_LIBS)

# Build the debug comparison library
$(DEBUG_TARGET): $(DEBUG_SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(DEBUG_TARGET) $(DEBUG_SOURCE) $(CUDA_LIBS)

# Build the Pippenger-based CUDA MSM library
$(PIPPENGER_TARGET): $(PIPPENGER_SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(PIPPENGER_TARGET) $(PIPPENGER_SOURCE) $(CUDA_LIBS)

# Build the hybrid CPU/GPU library  
$(HYBRID_TARGET): $(HYBRID_SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(HYBRID_TARGET) $(HYBRID_SOURCE) $(CUDA_LIBS)

# Build the direct CPU call library (requires barretenberg linking)
$(DIRECT_TARGET): $(DIRECT_SOURCE)
	$(NVCC) $(NVCC_FLAGS) -I../../.. -o $(DIRECT_TARGET) $(DIRECT_SOURCE) $(CUDA_LIBS)

# Build the GPU affine acceleration library
$(ACCELERATION_TARGET): $(ACCELERATION_SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(ACCELERATION_TARGET) $(ACCELERATION_SOURCE) $(CUDA_LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(DEBUG_TARGET) $(PIPPENGER_TARGET) $(HYBRID_TARGET) $(DIRECT_TARGET) $(ACCELERATION_TARGET)

# Install to common locations for testing
install: $(TARGET) $(DEBUG_TARGET) $(PIPPENGER_TARGET) $(HYBRID_TARGET) $(DIRECT_TARGET) $(ACCELERATION_TARGET)
	cp $(TARGET) .
	cp $(DEBUG_TARGET) .
	cp $(PIPPENGER_TARGET) .
	cp $(HYBRID_TARGET) .
	cp $(DIRECT_TARGET) .
	cp $(ACCELERATION_TARGET) .
	cp $(TARGET) /home/puckapao/ 2>/dev/null || true
	cp $(DEBUG_TARGET) /home/puckapao/ 2>/dev/null || true
	cp $(PIPPENGER_TARGET) /home/puckapao/ 2>/dev/null || true
	cp $(HYBRID_TARGET) /home/puckapao/ 2>/dev/null || true
	cp $(DIRECT_TARGET) /home/puckapao/ 2>/dev/null || true
	cp $(ACCELERATION_TARGET) /home/puckapao/ 2>/dev/null || true
	cp $(TARGET) ../../ 2>/dev/null || true
	cp $(DEBUG_TARGET) ../../ 2>/dev/null || true
	cp $(PIPPENGER_TARGET) ../../ 2>/dev/null || true
	cp $(HYBRID_TARGET) ../../ 2>/dev/null || true
	cp $(DIRECT_TARGET) ../../ 2>/dev/null || true
	cp $(ACCELERATION_TARGET) ../../ 2>/dev/null || true

.PHONY: all clean install
