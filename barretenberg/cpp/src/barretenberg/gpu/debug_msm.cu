// debug_msm.cu - Parallel CPU/GPU MSM comparison with detailed logging
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

// -------------------------------
// BN254 Constants (same as GPU)
// -------------------------------

// BN254 Fq modulus (base field)
__constant__ uint64_t BN254_FQ_MOD[4] = {
    0x3c208c16d87cfd47ULL,
    0x97816a916871ca8dULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};

__constant__ uint64_t BN254_FQ_R2[4] = {
    0xf32cfc5b538afa89ULL,
    0xb5e71911d44501fbULL,
    0x47ab1eff0a417ff6ULL,
    0x06d89f71cab8351fULL
};

__constant__ uint64_t BN254_FQ_NINV = 0x87d20782e4866389ULL;

static const uint64_t BN254_FQ_MOD_HOST[4] = {
    0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};

// -------------------------------
// Data Structures
// -------------------------------

struct FieldElement {
    uint64_t limbs[4];
    
    void print(const char* name) const {
        std::cout << name << ": " << std::hex << std::setfill('0')
                  << std::setw(16) << limbs[3] << "_"
                  << std::setw(16) << limbs[2] << "_"
                  << std::setw(16) << limbs[1] << "_"
                  << std::setw(16) << limbs[0] << std::dec << std::endl;
    }
    
    bool equals(const FieldElement& other) const {
        return (limbs[0] == other.limbs[0]) && (limbs[1] == other.limbs[1]) && 
               (limbs[2] == other.limbs[2]) && (limbs[3] == other.limbs[3]);
    }
    
    bool is_zero() const {
        return (limbs[0] | limbs[1] | limbs[2] | limbs[3]) == 0;
    }
};

struct AffinePoint {
    FieldElement x, y;
    
    void print(const char* name) const {
        std::cout << name << ":" << std::endl;
        x.print("  x");
        y.print("  y");
    }
    
    bool equals(const AffinePoint& other) const {
        return x.equals(other.x) && y.equals(other.y);
    }
};

struct JacobianPoint {
    FieldElement X, Y, Z;
    
    void print(const char* name) const {
        std::cout << name << ":" << std::endl;
        X.print("  X");
        Y.print("  Y");
        Z.print("  Z");
    }
    
    bool is_infinity() const {
        return Z.is_zero();
    }
};

// -------------------------------
// Host Field Operations (for CPU comparison)
// -------------------------------

static void host_fq_add(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t sum = (__uint128_t)a->limbs[i] + b->limbs[i] + carry;
        result->limbs[i] = (uint64_t)sum;
        carry = sum >> 64;
    }
    
    // Check if >= modulus and subtract if needed
    bool ge_mod = carry;
    if (!ge_mod) {
        for (int i = 3; i >= 0; i--) {
            if (result->limbs[i] > BN254_FQ_MOD_HOST[i]) {
                ge_mod = true;
                break;
            } else if (result->limbs[i] < BN254_FQ_MOD_HOST[i]) {
                break;
            }
        }
    }
    
    if (ge_mod) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            __uint128_t diff = (__uint128_t)result->limbs[i] - BN254_FQ_MOD_HOST[i] - borrow;
            result->limbs[i] = (uint64_t)diff;
            borrow = (diff >> 127) & 1;
        }
    }
}

// Basic CPU scalar multiplication (naive double-and-add)
static void cpu_scalar_mul(AffinePoint* result, const AffinePoint* point, const FieldElement* scalar, int debug_index) {
    std::cout << "\n=== CPU Scalar Mul " << debug_index << " ===" << std::endl;
    point->print("Input point");
    scalar->print("Input scalar");
    
    // For now, just copy the point (simplified)
    // This is where we'd implement proper CPU scalar multiplication
    *result = *point;
    
    std::cout << "CPU Scalar Mul " << debug_index << " completed" << std::endl;
    result->print("CPU Result");
}

// CPU MSM using naive approach (sum of scalar multiplications)
static void cpu_naive_msm(AffinePoint* result, const AffinePoint* points, const FieldElement* scalars, size_t num_points) {
    std::cout << "\n=== CPU NAIVE MSM START ===" << std::endl;
    std::cout << "Processing " << num_points << " points" << std::endl;
    
    // Initialize accumulator to point at infinity (simplified)
    AffinePoint accumulator;
    accumulator.x = {{0, 0, 0, 0}};
    accumulator.y = {{0, 0, 0, 0}};
    bool accumulator_is_infinity = true;
    
    // Process first few points for debugging
    size_t debug_limit = std::min((size_t)5, num_points);
    
    for (size_t i = 0; i < debug_limit; i++) {
        if (!scalars[i].is_zero()) {
            AffinePoint point_result;
            cpu_scalar_mul(&point_result, &points[i], &scalars[i], i);
            
            if (accumulator_is_infinity) {
                accumulator = point_result;
                accumulator_is_infinity = false;
                std::cout << "First non-zero result becomes accumulator" << std::endl;
                accumulator.print("Accumulator");
            } else {
                // Add point_result to accumulator (simplified - just keep first)
                std::cout << "Adding to accumulator (simplified)" << std::endl;
            }
        }
    }
    
    *result = accumulator;
    std::cout << "=== CPU NAIVE MSM END ===" << std::endl;
    result->print("CPU Final Result");
}

// -------------------------------
// GPU Debugging Kernel
// -------------------------------

__global__ void debug_msm_kernel(
    AffinePoint* d_gpu_results,
    const AffinePoint* d_points,
    const FieldElement* d_scalars,
    size_t num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // Only debug first few points
    if (idx >= 5) {
        // Set to infinity for points we're not debugging
        d_gpu_results[idx].x.limbs[0] = 0;
        d_gpu_results[idx].x.limbs[1] = 0;
        d_gpu_results[idx].x.limbs[2] = 0;
        d_gpu_results[idx].x.limbs[3] = 0;
        d_gpu_results[idx].y.limbs[0] = 0;
        d_gpu_results[idx].y.limbs[1] = 0;
        d_gpu_results[idx].y.limbs[2] = 0;
        d_gpu_results[idx].y.limbs[3] = 0;
        return;
    }
    
    const AffinePoint* point = &d_points[idx];
    const FieldElement* scalar = &d_scalars[idx];
    
    // Debug print (will be visible on first thread)
    if (idx == 0) {
        printf("\n=== GPU Debug Point %d ===\n", idx);
        printf("Point X: %016llx_%016llx_%016llx_%016llx\n",
               point->x.limbs[3], point->x.limbs[2], point->x.limbs[1], point->x.limbs[0]);
        printf("Scalar: %016llx_%016llx_%016llx_%016llx\n",
               scalar->limbs[3], scalar->limbs[2], scalar->limbs[1], scalar->limbs[0]);
    }
    
    // For now, simplified: just copy the input point
    d_gpu_results[idx] = *point;
    
    if (idx == 0) {
        printf("GPU Point %d completed\n", idx);
    }
}

// -------------------------------
// Debug Interface
// -------------------------------

extern "C" int debug_msm_compare(
    void* cpu_result_ptr,
    void* gpu_result_ptr,
    const void* points_ptr,
    const void* scalars_ptr,
    size_t num_points
) {
    std::cout << "\n=== DEBUG MSM COMPARISON START ===" << std::endl;
    std::cout << "Comparing CPU vs GPU MSM with " << num_points << " points" << std::endl;
    
    const AffinePoint* h_points = (const AffinePoint*)points_ptr;
    const FieldElement* h_scalars = (const FieldElement*)scalars_ptr;
    
    // Show first few input points and scalars
    std::cout << "\n=== INPUT DATA ===" << std::endl;
    size_t show_limit = std::min((size_t)3, num_points);
    for (size_t i = 0; i < show_limit; i++) {
        std::cout << "\nInput " << i << ":" << std::endl;
        h_points[i].print("Point");
        h_scalars[i].print("Scalar");
    }
    
    // Run CPU version
    auto cpu_start = std::chrono::high_resolution_clock::now();
    AffinePoint cpu_result;
    cpu_naive_msm(&cpu_result, h_points, h_scalars, num_points);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    // Run GPU version
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    AffinePoint* d_points;
    FieldElement* d_scalars;
    AffinePoint* d_results;
    
    cudaMalloc(&d_points, num_points * sizeof(AffinePoint));
    cudaMalloc(&d_scalars, num_points * sizeof(FieldElement));
    cudaMalloc(&d_results, num_points * sizeof(AffinePoint));
    
    cudaMemcpy(d_points, h_points, num_points * sizeof(AffinePoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalars, h_scalars, num_points * sizeof(FieldElement), cudaMemcpyHostToDevice);
    
    // Launch debug kernel
    int block_size = 256;
    int num_blocks = (num_points + block_size - 1) / block_size;
    debug_msm_kernel<<<num_blocks, block_size>>>(d_results, d_points, d_scalars, num_points);
    cudaDeviceSynchronize();
    
    // Copy results back
    std::vector<AffinePoint> gpu_individual_results(num_points);
    cudaMemcpy(gpu_individual_results.data(), d_results, num_points * sizeof(AffinePoint), cudaMemcpyDeviceToHost);
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    
    // Compare results
    std::cout << "\n=== COMPARISON RESULTS ===" << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << " μs" << std::endl;
    std::cout << "GPU time: " << gpu_time.count() << " μs" << std::endl;
    
    std::cout << "\n=== INDIVIDUAL SCALAR MULTIPLICATIONS ===" << std::endl;
    for (size_t i = 0; i < show_limit; i++) {
        std::cout << "\nComparison " << i << ":" << std::endl;
        std::cout << "CPU would compute: [point " << i << "] * [scalar " << i << "]" << std::endl;
        std::cout << "GPU computed:" << std::endl;
        gpu_individual_results[i].print("GPU result");
        
        if (gpu_individual_results[i].equals(h_points[i])) {
            std::cout << "✓ GPU correctly copied input point" << std::endl;
        } else {
            std::cout << "✗ GPU result differs from input point" << std::endl;
        }
    }
    
    // For now, just set results to show the comparison
    *(AffinePoint*)cpu_result_ptr = cpu_result;
    *(AffinePoint*)gpu_result_ptr = gpu_individual_results[0]; // Use first GPU result
    
    // Cleanup
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_results);
    
    std::cout << "\n=== DEBUG MSM COMPARISON END ===" << std::endl;
    return 0;
}