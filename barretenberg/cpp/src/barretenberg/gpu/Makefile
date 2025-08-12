# Separate Makefile to compile CUDA library
NVCC = nvcc
CUDA_FLAGS = -shared -Xcompiler -fPIC -O3

libcuda_msm.so: cuda_msm.cu
	$(NVCC) $(CUDA_FLAGS) -o libcuda_msm.so cuda_msm.cu

clean:
	rm -f libcuda_msm.so

.PHONY: clean
