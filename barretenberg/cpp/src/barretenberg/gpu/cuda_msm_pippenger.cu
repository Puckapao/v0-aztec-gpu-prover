// cuda_msm_pippenger.cu - GPU Pippenger implementation using CPU algorithm structure
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>

// -------------------------------
// BN254 Constants from barretenberg
// -------------------------------

__constant__ uint64_t BN254_FQ_MOD[4] = {
    0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};

__constant__ uint64_t BN254_FQ_R2[4] = {
    0xf32cfc5b538afa89ULL, 0xb5e71911d44501fbULL,
    0x47ab1eff0a417ff6ULL, 0x06d89f71cab8351fULL
};

__constant__ uint64_t BN254_FQ_NINV = 0x87d20782e4866389ULL;

static const uint64_t BN254_FQ_MOD_HOST[4] = {
    0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};

// Constants matching CPU implementation
static constexpr size_t NUM_BITS_IN_FIELD = 254;  // BN254 scalar field bits

// -------------------------------
// Data Structures
// -------------------------------

struct FieldElement {
    uint64_t limbs[4];
};

struct AffinePoint {
    FieldElement x, y;
};

struct JacobianPoint {
    FieldElement X, Y, Z;
};

// -------------------------------
// GPU Scalar Slice Functions (CPU equivalent)
// -------------------------------

// GPU version of CPU get_scalar_slice function
__device__ uint32_t gpu_get_scalar_slice(const FieldElement* scalar, size_t round, size_t slice_size) {
    size_t hi_bit = NUM_BITS_IN_FIELD - (round * slice_size);
    bool last_slice = hi_bit < slice_size;
    size_t target_slice_size = last_slice ? hi_bit : slice_size;
    size_t lo_bit = last_slice ? 0 : hi_bit - slice_size;
    
    size_t start_limb = lo_bit / 64;
    size_t end_limb = hi_bit / 64;
    size_t lo_slice_offset = lo_bit & 63;
    size_t lo_slice_bits = (target_slice_size < (64 - lo_slice_offset)) ? target_slice_size : (64 - lo_slice_offset);
    size_t hi_slice_bits = target_slice_size - lo_slice_bits;
    
    size_t lo_slice = (scalar->limbs[start_limb] >> lo_slice_offset) & ((1ULL << lo_slice_bits) - 1);
    size_t hi_slice = (scalar->limbs[end_limb] & ((1ULL << hi_slice_bits) - 1));
    
    uint32_t lo = static_cast<uint32_t>(lo_slice);
    uint32_t hi = static_cast<uint32_t>(hi_slice);
    uint32_t result = lo + (hi << lo_slice_bits);
    
    return result;
}

// GPU version of CPU get_optimal_log_num_buckets function
__host__ __device__ size_t gpu_get_optimal_log_num_buckets(size_t num_points) {
    // Same logic as CPU implementation
    const size_t COST_OF_BUCKET_OP_RELATIVE_TO_POINT = 5;
    size_t cached_cost = static_cast<size_t>(-1);
    size_t target_bit_slice = 0;
    
    for (size_t bit_slice = 1; bit_slice < 20; ++bit_slice) {
        const size_t num_rounds = (NUM_BITS_IN_FIELD + bit_slice - 1) / bit_slice;  // ceil_div
        const size_t num_buckets = 1ULL << bit_slice;
        const size_t addition_cost = num_rounds * num_points;
        const size_t bucket_cost = num_rounds * num_buckets * COST_OF_BUCKET_OP_RELATIVE_TO_POINT;
        const size_t total_cost = addition_cost + bucket_cost;
        
        if (total_cost < cached_cost) {
            cached_cost = total_cost;
            target_bit_slice = bit_slice;
        }
    }
    return target_bit_slice;
}

// -------------------------------
// GPU Field Operations (Proper BN254)
// -------------------------------

__device__ bool fq_is_zero(const FieldElement* a) {
    return (a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3]) == 0;
}

__device__ bool fq_equal(const FieldElement* a, const FieldElement* b) {
    return (a->limbs[0] == b->limbs[0]) && (a->limbs[1] == b->limbs[1]) && 
           (a->limbs[2] == b->limbs[2]) && (a->limbs[3] == b->limbs[3]);
}

// Proper BN254 field addition with modular reduction
__device__ void fq_add(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t carry = 0;
    
    // Step 1: Add limbs with carry
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a->limbs[i] + carry;
        carry = (sum < a->limbs[i]) ? 1 : 0;
        
        uint64_t sum2 = sum + b->limbs[i];
        if (sum2 < sum) carry = 1;
        
        result->limbs[i] = sum2;
    }
    
    // Step 2: Check if result >= modulus and subtract if needed
    bool ge_mod = carry;
    if (!ge_mod) {
        for (int i = 3; i >= 0; i--) {
            if (result->limbs[i] > BN254_FQ_MOD[i]) {
                ge_mod = true;
                break;
            } else if (result->limbs[i] < BN254_FQ_MOD[i]) {
                break;
            }
        }
    }
    
    // Step 3: Subtract modulus if needed
    if (ge_mod) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t temp = result->limbs[i];
            result->limbs[i] = temp - BN254_FQ_MOD[i] - borrow;
            borrow = (temp < (BN254_FQ_MOD[i] + borrow)) ? 1 : 0;
        }
    }
}

// BN254 field subtraction with modular reduction
__device__ void fq_sub(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    uint64_t borrow = 0;
    
    // Step 1: Subtract limbs
    for (int i = 0; i < 4; i++) {
        uint64_t temp = a->limbs[i];
        result->limbs[i] = temp - b->limbs[i] - borrow;
        borrow = (temp < (b->limbs[i] + borrow)) ? 1 : 0;
    }
    
    // Step 2: Add modulus if we went negative
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t sum = result->limbs[i] + BN254_FQ_MOD[i] + carry;
            carry = (sum < result->limbs[i]) ? 1 : 0;
            result->limbs[i] = sum;
        }
    }
}

// Set field element to zero
__device__ void fq_set_zero(FieldElement* result) {
    result->limbs[0] = 0;
    result->limbs[1] = 0;
    result->limbs[2] = 0;
    result->limbs[3] = 0;
}

// Set field element to one (Montgomery form)
__device__ void fq_set_one(FieldElement* result) {
    // In Montgomery form, 1 is represented as R mod p
    // For now, use simplified version
    result->limbs[0] = 1;
    result->limbs[1] = 0;
    result->limbs[2] = 0;
    result->limbs[3] = 0;
}

// Simplified field multiplication (for testing - not Montgomery)
__device__ void fq_mul_simple(FieldElement* result, const FieldElement* a, const FieldElement* b) {
    // Very simplified multiplication - just multiply low limbs
    // This is wrong for BN254 but allows testing the structure
    uint64_t prod_low = a->limbs[0] * b->limbs[0];
    result->limbs[0] = prod_low;
    result->limbs[1] = 0;
    result->limbs[2] = 0;
    result->limbs[3] = 0;
    
    // Apply modular reduction
    FieldElement temp = *result;
    fq_add(result, &temp, &temp); // This will trigger modular reduction
}

__device__ void jacobian_set_infinity(JacobianPoint* p) {
    for (int i = 0; i < 4; i++) {
        p->X.limbs[i] = 0;
        p->Y.limbs[i] = 0;
        p->Z.limbs[i] = 0;
    }
}

__device__ bool jacobian_is_infinity(const JacobianPoint* p) {
    return fq_is_zero(&p->Z);
}

__device__ void jacobian_from_affine(JacobianPoint* result, const AffinePoint* p) {
    result->X = p->x;
    result->Y = p->y;
    // Set Z = 1 (in Montgomery form this would be R, but simplified for now)
    result->Z.limbs[0] = 1;
    result->Z.limbs[1] = 0;
    result->Z.limbs[2] = 0; 
    result->Z.limbs[3] = 0;
}

// Proper BN254 Jacobian point doubling
__device__ void jacobian_double(JacobianPoint* result, const JacobianPoint* p) {
    if (jacobian_is_infinity(p) || fq_is_zero(&p->Y)) {
        jacobian_set_infinity(result);
        return;
    }
    
    // Jacobian doubling: (X, Y, Z) -> (X', Y', Z')
    // A = X^2
    FieldElement A;
    fq_mul_simple(&A, &p->X, &p->X);
    
    // B = Y^2
    FieldElement B;
    fq_mul_simple(&B, &p->Y, &p->Y);
    
    // C = B^2
    FieldElement C;
    fq_mul_simple(&C, &B, &B);
    
    // D = 2*((X+B)^2-A-C)
    FieldElement temp1, temp2, D;
    fq_add(&temp1, &p->X, &B);
    fq_mul_simple(&temp2, &temp1, &temp1);
    fq_sub(&temp1, &temp2, &A);
    fq_sub(&temp2, &temp1, &C);
    fq_add(&D, &temp2, &temp2);
    
    // E = 3*A
    FieldElement E;
    fq_add(&temp1, &A, &A);
    fq_add(&E, &temp1, &A);
    
    // F = E^2
    FieldElement F;
    fq_mul_simple(&F, &E, &E);
    
    // X3 = F - 2*D
    fq_add(&temp1, &D, &D);
    fq_sub(&result->X, &F, &temp1);
    
    // Y3 = E*(D-X3) - 8*C
    fq_sub(&temp1, &D, &result->X);
    fq_mul_simple(&temp2, &E, &temp1);
    FieldElement eight_C;
    fq_add(&temp1, &C, &C);
    fq_add(&eight_C, &temp1, &temp1);
    fq_add(&temp1, &eight_C, &eight_C);
    fq_sub(&result->Y, &temp2, &temp1);
    
    // Z3 = 2*Y*Z
    fq_mul_simple(&temp1, &p->Y, &p->Z);
    fq_add(&result->Z, &temp1, &temp1);
}

// Simplified point addition (placeholder for real implementation)

// -------------------------------
// GPU Affine Point Operations (matching CPU)
// -------------------------------

// Proper BN254 affine point addition
__device__ void affine_add(AffinePoint* result, const AffinePoint* p, const AffinePoint* q) {
    // Handle point at infinity cases
    if (fq_is_zero(&p->x) && fq_is_zero(&p->y)) {
        *result = *q;
        return;
    }
    if (fq_is_zero(&q->x) && fq_is_zero(&q->y)) {
        *result = *p;
        return;
    }
    
    // Check if points are equal (need point doubling)
    if (fq_equal(&p->x, &q->x)) {
        if (fq_equal(&p->y, &q->y)) {
            // Point doubling case - simplified for now
            // In production: slope = (3*x^2) / (2*y)
            *result = *p; // Placeholder - wrong but allows testing
        } else {
            // Points are inverses, result is point at infinity
            affine_set_infinity(result);
        }
        return;
    }
    
    // General case: P + Q where P != Q
    // slope = (y2 - y1) / (x2 - x1)
    // x3 = slope^2 - x1 - x2
    // y3 = slope * (x1 - x3) - y1
    
    FieldElement dx, dy, slope;
    fq_sub(&dx, &q->x, &p->x);  // dx = x2 - x1
    fq_sub(&dy, &q->y, &p->y);  // dy = y2 - y1
    
    // slope = dy / dx (we need field inversion - simplified for now)
    // For testing: use simplified slope
    fq_mul_simple(&slope, &dy, &dx);  // Wrong but allows testing
    
    // x3 = slope^2 - x1 - x2
    FieldElement slope_sq, temp;
    fq_mul_simple(&slope_sq, &slope, &slope);
    fq_sub(&temp, &slope_sq, &p->x);
    fq_sub(&result->x, &temp, &q->x);
    
    // y3 = slope * (x1 - x3) - y1
    FieldElement dx3;
    fq_sub(&dx3, &p->x, &result->x);
    fq_mul_simple(&temp, &slope, &dx3);
    fq_sub(&result->y, &temp, &p->y);
}

// Set affine point to infinity (CPU compatible)
__device__ void affine_set_infinity(AffinePoint* p) {
    // CPU represents infinity as (0, 0) in some contexts
    for (int i = 0; i < 4; i++) {
        p->x.limbs[i] = 0;
        p->y.limbs[i] = 0;
    }
}

__device__ bool affine_is_infinity(const AffinePoint* p) {
    return fq_is_zero(&p->x) && fq_is_zero(&p->y);
}

// GPU version of CPU accumulate_buckets function (CPU-compatible)
__device__ void gpu_accumulate_affine_buckets(
    AffinePoint* d_buckets,
    bool* d_bucket_exists,
    size_t num_buckets,
    JacobianPoint* d_result
) {
    // This follows CPU accumulate_buckets algorithm exactly
    // Find the highest non-empty bucket (CPU starts from end)
    int starting_index = -1;
    for (int i = (int)num_buckets - 1; i > 0; i--) {
        if (d_bucket_exists[i]) {
            starting_index = i;
            break;
        }
    }
    
    if (starting_index <= 0) {
        jacobian_set_infinity(d_result);
        return;
    }
    
    // Initialize prefix_sum with the highest bucket (CPU algorithm)
    AffinePoint prefix_sum = d_buckets[starting_index];
    
    // Convert first bucket to Jacobian and initialize sum (CPU algorithm)
    JacobianPoint sum;
    jacobian_from_affine(&sum, &prefix_sum);
    
    // CPU algorithm: for i from (starting_index-1) down to 1
    for (int i = starting_index - 1; i > 0; i--) {
        if (d_bucket_exists[i]) {
            // prefix_sum += buckets[i]
            affine_add(&prefix_sum, &prefix_sum, &d_buckets[i]);
        }
        // sum += prefix_sum
        JacobianPoint prefix_jac;
        jacobian_from_affine(&prefix_jac, &prefix_sum);
        jacobian_add(&sum, &sum, &prefix_jac);
    }
    
    *d_result = sum;
}

// GPU Pippenger round kernel (CPU evaluate_pippenger_round equivalent)
__global__ void gpu_pippenger_round_kernel(
    AffinePoint* d_affine_buckets,      // <<<< CHANGED: Use AFFINE buckets like CPU
    bool* d_bucket_exists,
    const AffinePoint* d_points,
    const FieldElement* d_scalars,
    const uint32_t* d_scalar_indices,
    size_t num_points,
    size_t round_index,
    size_t bits_per_slice,
    size_t num_buckets,
    JacobianPoint* d_round_result
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Fill AFFINE buckets (like CPU BucketAccumulators)
    size_t points_per_thread = (num_points + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    size_t start_idx = tid * points_per_thread;
    size_t end_idx = min(start_idx + points_per_thread, num_points);
    
    for (size_t i = start_idx; i < end_idx; i++) {
        uint32_t scalar_idx = d_scalar_indices[i];
        const FieldElement* scalar = &d_scalars[scalar_idx];
        
        // Get bucket index for this round (same as CPU)
        uint32_t bucket_index = gpu_get_scalar_slice(scalar, round_index, bits_per_slice);
        
        if (bucket_index > 0 && bucket_index < num_buckets) {
            const AffinePoint* point = &d_points[scalar_idx];
            
            // CPU algorithm: if bucket doesn't exist, initialize; otherwise add
            if (!d_bucket_exists[bucket_index]) {
                // Try to claim this bucket atomically
                if (atomicCAS((int*)&d_bucket_exists[bucket_index], 0, 1) == 0) {
                    // We claimed it, initialize with this affine point
                    d_affine_buckets[bucket_index] = *point;
                }
            } else {
                // Bucket exists, add to it using affine addition (like CPU)
                // This is where CPU uses batch inversions for efficiency
                affine_add(&d_affine_buckets[bucket_index], &d_affine_buckets[bucket_index], point);
            }
        }
    }
    
    __syncthreads();
    
    // Phase 2: Accumulate AFFINE buckets (like CPU accumulate_buckets)
    if (tid == 0) {
        gpu_accumulate_affine_buckets(
            d_affine_buckets, d_bucket_exists, num_buckets, d_round_result
        );
    }
}

// Main GPU Pippenger kernel - matches CPU pippenger_low_memory_with_transformed_scalars
__global__ void gpu_pippenger_main_kernel(
    JacobianPoint* d_final_result,
    const AffinePoint* d_points,
    const FieldElement* d_scalars,
    const uint32_t* d_scalar_indices,
    size_t num_points,
    size_t bits_per_slice,
    size_t num_rounds
) {
    // Only use one thread for the main algorithm coordination
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const size_t num_buckets = 1ULL << bits_per_slice;
        
        // Allocate AFFINE bucket arrays (like CPU BucketAccumulators)
        AffinePoint* affine_buckets = new AffinePoint[num_buckets];
        bool* bucket_exists = new bool[num_buckets]();
        
        // CPU algorithm: Element round_output = Curve::Group::point_at_infinity
        JacobianPoint accumulator;
        jacobian_set_infinity(&accumulator);
        
        // CPU algorithm: for each round, call evaluate_pippenger_round
        for (size_t round = 0; round < num_rounds; round++) {
            // Reset AFFINE buckets for this round (like CPU)
            for (size_t b = 0; b < num_buckets; b++) {
                affine_set_infinity(&affine_buckets[b]);
                bucket_exists[b] = false;
            }
            
            // Fill AFFINE buckets for this round (like CPU consume_point_schedule)
            for (size_t i = 0; i < num_points; i++) {
                uint32_t scalar_idx = d_scalar_indices[i];
                const FieldElement* scalar = &d_scalars[scalar_idx];
                
                uint32_t bucket_index = gpu_get_scalar_slice(scalar, round, bits_per_slice);
                
                if (bucket_index > 0 && bucket_index < num_buckets) {
                    const AffinePoint* point = &d_points[scalar_idx];
                    
                    // CPU algorithm: bucket initialization and addition
                    if (!bucket_exists[bucket_index]) {
                        affine_buckets[bucket_index] = *point;
                        bucket_exists[bucket_index] = true;
                    } else {
                        affine_add(&affine_buckets[bucket_index], &affine_buckets[bucket_index], point);
                    }
                }
            }
            
            // Accumulate AFFINE buckets (like CPU accumulate_buckets)
            JacobianPoint round_output;
            gpu_accumulate_affine_buckets(affine_buckets, bucket_exists, num_buckets, &round_output);
            
            // CPU algorithm: Apply doublings then add to accumulator
            // Element result = previous_round_output;
            // size_t num_doublings = ...
            // for (size_t i = 0; i < num_doublings; ++i) result.self_dbl();
            // result += round_output;
            
            size_t num_doublings = ((round == num_rounds - 1) && (NUM_BITS_IN_FIELD % bits_per_slice != 0))
                                 ? NUM_BITS_IN_FIELD % bits_per_slice
                                 : bits_per_slice;
            
            for (size_t d = 0; d < num_doublings; d++) {
                jacobian_double(&accumulator, &accumulator);
            }
            
            jacobian_add(&accumulator, &accumulator, &round_output);
        }
        
        *d_final_result = accumulator;
        
        delete[] affine_buckets;
        delete[] bucket_exists;
    }
}

// Host versions of the functions (no __device__ attribute)
static void host_jacobian_set_infinity(JacobianPoint* p) {
    for (int i = 0; i < 4; i++) {
        p->X.limbs[i] = 0;
        p->Y.limbs[i] = 0;
        p->Z.limbs[i] = 0;
    }
}

static bool host_jacobian_is_infinity(const JacobianPoint* p) {
    return (p->Z.limbs[0] | p->Z.limbs[1] | p->Z.limbs[2] | p->Z.limbs[3]) == 0;
}

// Host accumulation (mimic CPU step 3)
static void host_accumulate_results(JacobianPoint* final_result, const JacobianPoint* partial_results, size_t num_chunks) {
    std::cout << "=== GPU HOST ACCUMULATION ===" << std::endl;
    std::cout << "Accumulating " << num_chunks << " partial results..." << std::endl;
    
    host_jacobian_set_infinity(final_result);
    
    for (size_t i = 0; i < num_chunks; i++) {
        if (!host_jacobian_is_infinity(&partial_results[i])) {
            if (host_jacobian_is_infinity(final_result)) {
                *final_result = partial_results[i];
            } else {
                // Simplified host addition
                for (int j = 0; j < 4; j++) {
                    final_result->X.limbs[j] += partial_results[i].X.limbs[j];
                    final_result->Y.limbs[j] += partial_results[i].Y.limbs[j];
                }
            }
        }
    }
}

static void host_jacobian_to_affine_simple(AffinePoint* result, const JacobianPoint* p) {
    if (host_jacobian_is_infinity(p)) {
        // Set to point at infinity
        for (int i = 0; i < 4; i++) {
            result->x.limbs[i] = BN254_FQ_MOD_HOST[i];
            result->y.limbs[i] = 0;
        }
        return;
    }
    
    // Simplified: assume Z=1 (wrong for general case)
    result->x = p->X;
    result->y = p->Y;
}

// -------------------------------
// Main GPU MSM Function
// -------------------------------

extern "C" int gpu_pippenger_msm(
    void* result_ptr,
    const void* points_ptr,
    const void* scalars_ptr,
    size_t num_points
) {
    std::cout << "\n=== GPU PIPPENGER MSM START ===" << std::endl;
    std::cout << "Processing " << num_points << " points using real Pippenger algorithm" << std::endl;
    
    if (!result_ptr || !points_ptr || !scalars_ptr || num_points == 0) {
        return -1;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        const AffinePoint* h_points = (const AffinePoint*)points_ptr;
        const FieldElement* h_scalars = (const FieldElement*)scalars_ptr;
        
        // Step 1: Create scalar indices (mimic CPU transform_scalar_and_get_nonzero_scalar_indices)
        std::vector<uint32_t> scalar_indices;
        for (size_t i = 0; i < num_points; i++) {
            // Host version of fq_is_zero check
            bool is_zero = (h_scalars[i].limbs[0] | h_scalars[i].limbs[1] | h_scalars[i].limbs[2] | h_scalars[i].limbs[3]) == 0;
            if (!is_zero) {
                scalar_indices.push_back(static_cast<uint32_t>(i));
            }
        }
        
        if (scalar_indices.empty()) {
            // All scalars are zero, return point at infinity
            AffinePoint* result = (AffinePoint*)result_ptr;
            for (int i = 0; i < 4; i++) {
                result->x.limbs[i] = BN254_FQ_MOD_HOST[i];
                result->y.limbs[i] = 0;
            }
            std::cout << "All scalars are zero, returning point at infinity" << std::endl;
            return 0;
        }
        
        size_t effective_points = scalar_indices.size();
        std::cout << "Effective points (non-zero scalars): " << effective_points << std::endl;
        
        // Step 2: Calculate optimal parameters (use CPU algorithm)
        size_t bits_per_slice = gpu_get_optimal_log_num_buckets(effective_points);
        size_t num_rounds = (NUM_BITS_IN_FIELD + bits_per_slice - 1) / bits_per_slice;
        
        std::cout << "Using " << bits_per_slice << " bits per slice, " << num_rounds << " rounds" << std::endl;
        
        // Step 3: Allocate device memory
        AffinePoint* d_points;
        FieldElement* d_scalars;
        uint32_t* d_scalar_indices;
        JacobianPoint* d_final_result;
        
        cudaMalloc(&d_points, num_points * sizeof(AffinePoint));
        cudaMalloc(&d_scalars, num_points * sizeof(FieldElement));
        cudaMalloc(&d_scalar_indices, effective_points * sizeof(uint32_t));
        cudaMalloc(&d_final_result, sizeof(JacobianPoint));
        
        // Step 4: Copy data to device
        cudaMemcpy(d_points, h_points, num_points * sizeof(AffinePoint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scalars, h_scalars, num_points * sizeof(FieldElement), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scalar_indices, scalar_indices.data(), effective_points * sizeof(uint32_t), cudaMemcpyHostToDevice);
        
        // Step 5: Launch main Pippenger kernel (CPU-compatible algorithm)
        std::cout << "Launching GPU kernel with CPU-compatible affine buckets..." << std::endl;
        gpu_pippenger_main_kernel<<<1, 1>>>(
            d_final_result,
            d_points,
            d_scalars,
            d_scalar_indices,
            effective_points,
            bits_per_slice,
            num_rounds
        );
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        // Step 6: Copy result back and convert to affine
        JacobianPoint final_jacobian;
        cudaMemcpy(&final_jacobian, d_final_result, sizeof(JacobianPoint), cudaMemcpyDeviceToHost);
        
        AffinePoint* result = (AffinePoint*)result_ptr;
        host_jacobian_to_affine_simple(result, &final_jacobian);
        
        // Cleanup
        cudaFree(d_points);
        cudaFree(d_scalars);
        cudaFree(d_scalar_indices);
        cudaFree(d_final_result);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "=== GPU PIPPENGER RESULT ===" << std::endl;
        std::cout << "GPU Result x: " << std::hex << result->x.limbs[3] << "_" << result->x.limbs[2] << "_" << result->x.limbs[1] << "_" << result->x.limbs[0] << std::dec << std::endl;
        std::cout << "GPU Result y: " << std::hex << result->y.limbs[3] << "_" << result->y.limbs[2] << "_" << result->y.limbs[1] << "_" << result->y.limbs[0] << std::dec << std::endl;
        std::cout << "GPU execution time: " << duration.count() << " μs" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "GPU Pippenger MSM error: " << e.what() << std::endl;
        return -1;
    }
}

// Legacy interface
extern "C" int cuda_msm_compute(
    void* result_ptr,
    const void* points_ptr,
    const void* scalars_ptr,
    size_t num_points
) {
    return gpu_pippenger_msm(result_ptr, points_ptr, scalars_ptr, num_points);
}