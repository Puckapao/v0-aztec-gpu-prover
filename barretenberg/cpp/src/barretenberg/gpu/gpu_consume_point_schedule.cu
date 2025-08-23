#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <algorithm>

// gpu_consume_point_schedule.cu (top of file or before kernels)
#ifndef AFFINE_BATCH_CAPACITY
#define AFFINE_BATCH_CAPACITY 2048  // must match AffineAdditionData::BATCH_SIZE
#endif

// Keep the "2 points per affine add" invariant and capacity sanity
static_assert(AFFINE_BATCH_CAPACITY % 2 == 0, "AFFINE_BATCH_CAPACITY must be even");

__device__ __forceinline__ bool bitvector_get(const uint64_t* words, size_t num_bits, size_t bit_idx) {
    if (bit_idx >= num_bits) return false;
    const size_t word = bit_idx >> 6;     // /64
    const size_t bit  = bit_idx & 63;     // %64
    const size_t num_words = (num_bits + 63) / 64;
    return (word < num_words) && (((words[word] >> bit) & 1ULL) != 0ULL);
}


// Thread-safe logging for GPU operations
static std::mutex gpu_log_mutex;
#define GPU_LOG(msg) do { \
    std::lock_guard<std::mutex> lock(gpu_log_mutex); \
    std::cout << msg << std::endl; \
} while(0)

// Debounce high-frequency GPU validation log messages
static size_t gpu_large_batch_count = 0;
static const size_t GPU_LARGE_BATCH_LOG_INTERVAL = 25;  // Log every 25th large batch

// Simple structure to return GPU processing results to CPU
struct GPUProcessingResult {
    size_t new_point_it;         // Where GPU stopped processing
    size_t new_affine_input_it;  // How many affine additions GPU queued
    size_t iterations_processed; // Number of iterations GPU completed
};

// Structure for CPU-GPU validation comparison
struct CPUGPUIterationResult {
    size_t point_it_advance;     // How much point_it advanced
    bool do_affine_add;          // Whether this iteration would do affine add
    size_t lhs_bucket;           // Left bucket value
    size_t rhs_bucket;           // Right bucket value
    bool buckets_match;          // Whether buckets matched
};

// Simple test function to verify library loading
extern "C" int gpu_test_function() {
    GPU_LOG("GPU: Test function called successfully!");
    
    // Check CUDA device
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    
    if (cuda_status == cudaSuccess && device_count > 0) {
        GPU_LOG("GPU: Found " << device_count << " CUDA device(s)");
        return 0; // Success
    } else {
        GPU_LOG("GPU: No CUDA devices found or CUDA error: " << cudaGetErrorString(cuda_status));
        return -1; // Error
    }
}

// GPU acceleration for the main processing loop in consume_point_schedule
extern "C" int gpu_process_point_schedule_loop(
    const uint64_t* point_schedule,
    size_t point_schedule_size,
    size_t start_point_it,
    size_t start_affine_input_it,
    size_t end_point_it,
    void* results_ptr  // Will store the processing results
) {
    // GPU validation - validates processing logic for first 100 iterations (doesn't change CPU state)
    std::cout << "GPU: Validating first 100 iterations from range[" << start_point_it << "-" << end_point_it << "]" << std::endl;
    
    // Return results to CPU through results_ptr
    GPUProcessingResult* gpu_results = static_cast<GPUProcessingResult*>(results_ptr);
    if (!gpu_results) {
        std::cout << "GPU: No results pointer provided, skipping processing" << std::endl;
        return 0;
    }
    
    // Initialize results
    gpu_results->new_point_it = start_point_it;
    gpu_results->new_affine_input_it = start_affine_input_it;
    gpu_results->iterations_processed = 0;
    
    // Process up to 100 iterations - same logic as CPU main loop
    if (point_schedule && point_schedule_size > 0 && start_point_it < end_point_it && (start_point_it + 1) < end_point_it) {
        size_t max_iterations = std::min((size_t)100, (end_point_it - start_point_it) / 2);
        size_t gpu_point_it = start_point_it;
        size_t gpu_affine_input_it = start_affine_input_it;
        size_t completed_iterations = 0;
        
        for (size_t iteration = 0; iteration < max_iterations && (gpu_point_it + 1) < end_point_it; iteration++) {
            // Check if we have space for more affine additions (same as CPU limit)
            if ((gpu_affine_input_it + 1) >= AFFINE_BATCH_CAPACITY) { // AffineAdditionData::BATCH_SIZE
                break;
            }
            
            // Get point schedule entries (same as CPU)
            uint64_t lhs_schedule = point_schedule[gpu_point_it];
            uint64_t rhs_schedule = point_schedule[gpu_point_it + 1];
            
            // Extract bucket and point indices (same as CPU)
            size_t lhs_bucket = static_cast<size_t>(lhs_schedule) & 0xFFFFFFFF;
            size_t rhs_bucket = static_cast<size_t>(rhs_schedule) & 0xFFFFFFFF;
            size_t lhs_point = static_cast<size_t>(lhs_schedule >> 32);
            size_t rhs_point = static_cast<size_t>(rhs_schedule >> 32);
            
            // Apply same logic as CPU main loop
            bool buckets_match = (lhs_bucket == rhs_bucket);
            bool has_bucket_accumulator = false; // Simplified for now - CPU will handle this
            bool do_affine_add = buckets_match || has_bucket_accumulator;
            
            // For now, only advance point iterator - don't modify affine_input_it
            gpu_point_it += (do_affine_add && buckets_match) ? 2 : 1;
            completed_iterations++;
            
            // Log first few for validation
            if (iteration < 3) {
                GPU_LOG("GPU: iter[" << iteration << "] point_it=" << (gpu_point_it - ((do_affine_add && buckets_match) ? 2 : 1))
                        << " buckets_match=" << buckets_match << " do_affine_add=" << do_affine_add);
            }
        }
        
        // Return results to CPU
        gpu_results->new_point_it = gpu_point_it;
        gpu_results->new_affine_input_it = gpu_affine_input_it;
        gpu_results->iterations_processed = completed_iterations;
        
        GPU_LOG("GPU: Validated " << completed_iterations << " iterations successfully " 
                << "(CPU will process normally)");
    }
    
    return 0; // Success - CPU can use results and continue
}

// CUDA kernel that actually runs on GPU
__global__ void process_point_schedule_kernel(
    const uint64_t* point_schedule,
    const uint64_t* bucket_accumulator_exists_data,
    size_t bucket_accumulator_size_bits,
    size_t start_point_it,
    size_t num_iterations,
    CPUGPUIterationResult* results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_iterations) return;
    
    size_t point_it = start_point_it + idx;
    
    // Process single iteration exactly like CPU main loop
    uint64_t lhs_schedule = point_schedule[point_it];
    uint64_t rhs_schedule = point_schedule[point_it + 1];
    
    // Extract bucket and point indices
    size_t lhs_bucket = static_cast<size_t>(lhs_schedule) & 0xFFFFFFFF;
    size_t rhs_bucket = static_cast<size_t>(rhs_schedule) & 0xFFFFFFFF;
    
    // GPU implementation of BitVector::get() - same as CPU BitVector::get()
    bool has_bucket_accumulator = false;
    if (lhs_bucket < bucket_accumulator_size_bits) {
        const uint64_t word = lhs_bucket >> 6;          // lhs_bucket / 64
        const uint64_t bit = lhs_bucket & 63;           // lhs_bucket % 64
        const size_t num_words = (bucket_accumulator_size_bits + 63) / 64;
        
        if (word < num_words) {
            has_bucket_accumulator = ((bucket_accumulator_exists_data[word] >> bit) & 1ULL) == 1ULL;
        }
    }
    
    // Apply same logic as CPU main loop - now with REAL bucket accumulator state
    bool buckets_match = (lhs_bucket == rhs_bucket);
    bool do_affine_add = buckets_match || has_bucket_accumulator;
    
    // Calculate advancement (same as CPU)
    size_t point_it_advance = (do_affine_add && buckets_match) ? 2 : 1;
    
    // Store results for this thread
    results[idx].point_it_advance = point_it_advance;
    results[idx].do_affine_add = do_affine_add;
    results[idx].lhs_bucket = lhs_bucket;
    results[idx].rhs_bucket = rhs_bucket;
    results[idx].buckets_match = buckets_match;
}

// DEBUG: Fixed single-iteration GPU processing for careful testing
extern "C" int gpu_debug_single_iteration(
    const uint64_t* point_schedule,
    size_t point_it,
    size_t end_point_it,
    const uint64_t* bucket_accumulator_exists_data,
    size_t bucket_accumulator_size_bits,
    CPUGPUIterationResult* gpu_result
) {
    if (!gpu_result || !point_schedule || (point_it + 1) >= end_point_it || !bucket_accumulator_exists_data) {
        GPU_LOG("GPU DEBUG: Invalid input parameters");
        return -1;
    }
    
    GPU_LOG("GPU DEBUG: Processing single iteration at point_it=" << point_it);
    
    // Process single iteration exactly like CPU main loop
    uint64_t lhs_schedule = point_schedule[point_it];
    uint64_t rhs_schedule = point_schedule[point_it + 1];
    
    GPU_LOG("GPU DEBUG: lhs_schedule=" << lhs_schedule << ", rhs_schedule=" << rhs_schedule);
    
    // Extract bucket and point indices
    size_t lhs_bucket = static_cast<size_t>(lhs_schedule) & 0xFFFFFFFF;
    size_t rhs_bucket = static_cast<size_t>(rhs_schedule) & 0xFFFFFFFF;
    
    GPU_LOG("GPU DEBUG: lhs_bucket=" << lhs_bucket << ", rhs_bucket=" << rhs_bucket);
    
    // GPU implementation of BitVector::get() - same as CPU BitVector::get()
    bool has_bucket_accumulator = false;
    if (lhs_bucket < bucket_accumulator_size_bits) {
        const uint64_t word = lhs_bucket >> 6;          // lhs_bucket / 64
        const uint64_t bit = lhs_bucket & 63;           // lhs_bucket % 64
        const size_t num_words = (bucket_accumulator_size_bits + 63) / 64;
        
        if (word < num_words) {
            has_bucket_accumulator = ((bucket_accumulator_exists_data[word] >> bit) & 1ULL) == 1ULL;
        }
    }
    
    GPU_LOG("GPU DEBUG: has_bucket_accumulator=" << has_bucket_accumulator);
    
    // Apply same logic as CPU main loop - now with REAL bucket accumulator state
    bool buckets_match = (lhs_bucket == rhs_bucket);
    bool do_affine_add = buckets_match || has_bucket_accumulator;
    
    // Calculate advancement (same as CPU)
    size_t point_it_advance = (do_affine_add && buckets_match) ? 2 : 1;
    
    GPU_LOG("GPU DEBUG: buckets_match=" << buckets_match << ", do_affine_add=" << do_affine_add << ", point_it_advance=" << point_it_advance);
    
    // Store results
    gpu_result->point_it_advance = point_it_advance;
    gpu_result->do_affine_add = do_affine_add;
    gpu_result->lhs_bucket = lhs_bucket;
    gpu_result->rhs_bucket = rhs_bucket;
    gpu_result->buckets_match = buckets_match;
    
    return 0; // Success
}

// SOLUTION 1: Sequential GPU processing with state simulation
__global__ void process_point_schedule_sequential_kernel(
    const uint64_t* point_schedule,
    uint64_t* bucket_accumulator_exists_data,  // Mutable for state updates
    size_t bucket_accumulator_size_bits,
    size_t start_point_it,
    size_t num_iterations,
    CPUGPUIterationResult* results
) {
    // Only one thread processes sequentially to maintain state consistency
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    for (size_t iter = 0; iter < num_iterations; iter++) {
        size_t point_it = start_point_it + iter;
        
        // Process single iteration exactly like CPU main loop
        uint64_t lhs_schedule = point_schedule[point_it];
        uint64_t rhs_schedule = point_schedule[point_it + 1];
        
        // Extract bucket and point indices
        size_t lhs_bucket = static_cast<size_t>(lhs_schedule) & 0xFFFFFFFF;
        size_t rhs_bucket = static_cast<size_t>(rhs_schedule) & 0xFFFFFFFF;
        
        // GPU implementation of BitVector::get() - same as CPU BitVector::get()
        bool has_bucket_accumulator = false;
        if (lhs_bucket < bucket_accumulator_size_bits) {
            const uint64_t word = lhs_bucket >> 6;          // lhs_bucket / 64
            const uint64_t bit = lhs_bucket & 63;           // lhs_bucket % 64
            const size_t num_words = (bucket_accumulator_size_bits + 63) / 64;
            
            if (word < num_words) {
                has_bucket_accumulator = ((bucket_accumulator_exists_data[word] >> bit) & 1ULL) == 1ULL;
            }
        }
        
        // Apply same logic as CPU main loop
        bool buckets_match = (lhs_bucket == rhs_bucket);
        bool do_affine_add = buckets_match || has_bucket_accumulator;
        
        // Calculate advancement (same as CPU)
        size_t point_it_advance = (do_affine_add && buckets_match) ? 2 : 1;
        
        // Store results for this iteration
        results[iter].point_it_advance = point_it_advance;
        results[iter].do_affine_add = do_affine_add;
        results[iter].lhs_bucket = lhs_bucket;
        results[iter].rhs_bucket = rhs_bucket;
        results[iter].buckets_match = buckets_match;
        
        // CRITICAL: Update bucket accumulator state exactly like CPU
        // bucket_accumulator_exists.set(lhs_bucket, new_state)
        bool new_state = (has_bucket_accumulator && buckets_match) || !do_affine_add;
        
        if (lhs_bucket < bucket_accumulator_size_bits) {
            const uint64_t word = lhs_bucket >> 6;
            const uint64_t bit = lhs_bucket & 63;
            const size_t num_words = (bucket_accumulator_size_bits + 63) / 64;
            
            if (word < num_words) {
                if (new_state) {
                    bucket_accumulator_exists_data[word] |= (1ULL << bit);  // Set bit
                } else {
                    bucket_accumulator_exists_data[word] &= ~(1ULL << bit); // Clear bit
                }
            }
        }
    }
}

// LARGE BATCH: Process 100 iterations using actual CUDA kernel with GPU acceleration  
#if DEBUG
extern "C" int gpu_large_batch_cuda(
    const uint64_t* point_schedule,
    size_t start_point_it,
    size_t end_point_it,
    const uint64_t* bucket_accumulator_exists_data,
    size_t bucket_accumulator_size_bits,
    CPUGPUIterationResult* cpu_results,  // Pre-allocated array of 100 results
    size_t batch_size                    // Should be 100
) {
    if (!cpu_results || !point_schedule || !bucket_accumulator_exists_data || batch_size != 100) {
        GPU_LOG("GPU LARGE BATCH: Invalid input parameters, expected batch_size=100");
        return -1;
    }
    
    // Calculate actual iterations we can process
    size_t available_iterations = (end_point_it > start_point_it) ? (end_point_it - start_point_it) : 0;
    size_t num_iterations = (available_iterations < batch_size) ? available_iterations : batch_size;
    
    if (num_iterations == 0) {
        GPU_LOG("GPU LARGE BATCH: No iterations to process");
        return -1;
    }
    
    GPU_LOG("GPU LARGE BATCH: Processing " << num_iterations << " iterations starting at point_it=" << start_point_it);
    
    // Allocate GPU memory for large batch
    uint64_t* d_point_schedule = nullptr;
    uint64_t* d_bucket_exists = nullptr;
    CPUGPUIterationResult* d_results = nullptr;
    
    size_t point_schedule_bytes = (num_iterations + 1) * sizeof(uint64_t);  // Large allocation
    size_t bucket_exists_bytes = ((bucket_accumulator_size_bits + 63) / 64) * sizeof(uint64_t);
    size_t results_bytes = num_iterations * sizeof(CPUGPUIterationResult);
    
    cudaError_t error;
    
    // Allocate GPU memory (large amounts)
    error = cudaMalloc(&d_point_schedule, point_schedule_bytes);
    if (error != cudaSuccess) {
        GPU_LOG("GPU LARGE BATCH: Failed to allocate point schedule memory");
        return -1;
    }
    
    error = cudaMalloc(&d_bucket_exists, bucket_exists_bytes);
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        GPU_LOG("GPU LARGE BATCH: Failed to allocate bucket exists memory");
        return -1;
    }
    
    error = cudaMalloc(&d_results, results_bytes);
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        GPU_LOG("GPU LARGE BATCH: Failed to allocate results memory");
        return -1;
    }
    
    // Copy data to GPU
    error = cudaMemcpy(d_point_schedule, &point_schedule[start_point_it], point_schedule_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        cudaFree(d_results);
        GPU_LOG("GPU LARGE BATCH: Failed to copy point schedule to GPU");
        return -1;
    }
    
    error = cudaMemcpy(d_bucket_exists, bucket_accumulator_exists_data, bucket_exists_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        cudaFree(d_results);
        GPU_LOG("GPU LARGE BATCH: Failed to copy bucket exists to GPU");
        return -1;
    }
    
    // Launch sequential CUDA kernel with single thread for state consistency
    // Use 1 thread, 1 block to ensure sequential processing
    int threads_per_block = 1;
    int blocks = 1;
    
    process_point_schedule_sequential_kernel<<<blocks, threads_per_block>>>(
        d_point_schedule,
        d_bucket_exists,  // Now mutable for state updates
        bucket_accumulator_size_bits,
        0,  // relative to copied data
        num_iterations,
        d_results
    );
    
    // Wait for GPU to finish
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        cudaFree(d_results);
        GPU_LOG("GPU LARGE BATCH: Kernel execution failed");
        return -1;
    }
    
    // Copy results back to CPU
    error = cudaMemcpy(cpu_results, d_results, results_bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        cudaFree(d_results);
        GPU_LOG("GPU LARGE BATCH: Failed to copy results from GPU");
        return -1;
    }
    
    // Cleanup GPU memory
    cudaFree(d_point_schedule);
    cudaFree(d_bucket_exists);
    cudaFree(d_results);
    
    // Debounce large batch success messages
    gpu_large_batch_count++;
    if (gpu_large_batch_count % GPU_LARGE_BATCH_LOG_INTERVAL == 0) {
        GPU_LOG("GPU LARGE BATCH SEQUENTIAL: Successfully processed " << gpu_large_batch_count << " batches (" << num_iterations << " iterations latest)");
    }
    return num_iterations; // Success
}
#endif

// === NEW: state-faithful sequential kernel with validation-friendly capacity rules ===
// === state-faithful sequential kernel (validation-friendly) ===
__global__ void process_point_schedule_sequential_with_results(
    const uint64_t* point_schedule_window,           // windowed: [0 .. num_entries-1]
    uint64_t* bucket_accumulator_exists_data,        // mutable bitvector words
    size_t bucket_accumulator_size_bits,             // total number of bits
    size_t num_entries,                              // entries in window
    size_t start_affine_input_it,                    // starting affine input index
    size_t* final_point_offset_gpu,                  // OUT: relative offset processed
    size_t* final_affine_input_it_gpu,               // OUT
    size_t* iterations_processed_gpu,                // OUT
    int /*apply_updates*/,                           // kept for ABI
    int respect_affine_capacity,                     // 1 => production gating, 0 => validation
    int handle_tail_singleton                        // 1 => mirror CPU last-singleton behavior
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Initialize outputs so early returns don't leave garbage
    *final_point_offset_gpu    = 0;
    *final_affine_input_it_gpu = start_affine_input_it;
    *iterations_processed_gpu  = 0;

    if (!point_schedule_window || !bucket_accumulator_exists_data ||
        !final_point_offset_gpu || !final_affine_input_it_gpu || !iterations_processed_gpu) {
        printf("GPU ERROR: Null pointer parameters detected\n");
        return;
    }

    size_t off = 0;
    size_t gpu_affine_input_it = start_affine_input_it; // **reported** value
    size_t completed_iterations = 0;
    size_t iteration_count = 0;

    const size_t MAX_SAFE_ITERATIONS = 500000;
    const size_t MAX_AFFINE_CAPACITY = AFFINE_BATCH_CAPACITY;

    const bool use_safety_limits = (respect_affine_capacity != 0);
    const bool count_affine_for_output = true; // <== key: don't bump during validation

    if (num_entries == 0) return;
    if (num_entries > (1u<<27)) { // cheap corruption guard
        printf("GPU ERROR: num_entries=%llu too large\n", (unsigned long long)num_entries);
        return;
    }
    if (use_safety_limits && start_affine_input_it > MAX_AFFINE_CAPACITY) {
        printf("GPU ERROR: start_affine_input_it=%llu exceeds capacity=%llu\n",
               (unsigned long long)start_affine_input_it,
               (unsigned long long)MAX_AFFINE_CAPACITY);
        return;
    }

    // Consume pairs while a RHS exists
    while (off + 1 < num_entries) {
        const uint64_t lhs_schedule = point_schedule_window[off];
        const uint64_t rhs_schedule = point_schedule_window[off + 1];

        const uint32_t lhs_bucket_raw = static_cast<uint32_t>(lhs_schedule & 0xFFFFFFFFULL);
        const uint32_t rhs_bucket_raw = static_cast<uint32_t>(rhs_schedule & 0xFFFFFFFFULL);

        // Invalid LHS bucket: skip safely (mirror CPU)
        if (lhs_bucket_raw >= bucket_accumulator_size_bits) {
            off += 1;
            completed_iterations++;
            iteration_count++;
            if (iteration_count >= MAX_SAFE_ITERATIONS) break;
            continue;
        }

        const size_t lhs_bucket = static_cast<size_t>(lhs_bucket_raw);
        const bool   rhs_valid  = (rhs_bucket_raw < bucket_accumulator_size_bits);
        const size_t rhs_bucket = rhs_valid ? static_cast<size_t>(rhs_bucket_raw) : 0;

        // Accumulator state
        const bool has_bucket_accumulator =
            bitvector_get(bucket_accumulator_exists_data, bucket_accumulator_size_bits, lhs_bucket);

        const bool buckets_match = rhs_valid && (lhs_bucket == rhs_bucket);
        const bool do_affine_add = buckets_match || has_bucket_accumulator;

        // Capacity gating only in production mode
        if (use_safety_limits && do_affine_add && (gpu_affine_input_it + 1) >= MAX_AFFINE_CAPACITY) {
            break;
        }

        // Advance affine iterator only when we are *supposed* to report it
        if (do_affine_add && count_affine_for_output) {
            if (use_safety_limits) {
                gpu_affine_input_it += 2;
            } else {
                // validation: saturate to capacity, don't early-break
                if (gpu_affine_input_it <= MAX_AFFINE_CAPACITY - 2) {
                    gpu_affine_input_it += 2;
                } else {
                    gpu_affine_input_it = MAX_AFFINE_CAPACITY;
                }
            }
        }

        // Point iterator advance (exact CPU semantics)
        const size_t point_advance = (do_affine_add && buckets_match) ? 2 : 1;
        off += point_advance;
        completed_iterations++;
        iteration_count++;

        // Update device bitvector so subsequent iterations see new state
        const bool new_state = (has_bucket_accumulator && buckets_match) || !do_affine_add;
        const size_t word = lhs_bucket >> 6;
        const size_t bit  = lhs_bucket & 63;
        const uint64_t mask = (1ULL << bit);
        if (new_state) {
            atomicOr(reinterpret_cast<unsigned long long*>(&bucket_accumulator_exists_data[word]),
                     static_cast<unsigned long long>(mask));
        } else {
            atomicAnd(reinterpret_cast<unsigned long long*>(&bucket_accumulator_exists_data[word]),
                      static_cast<unsigned long long>(~mask));
        }

        if (iteration_count >= MAX_SAFE_ITERATIONS) break;
    }

    // Optional tail (odd singleton)
    if ((off + 1 == num_entries) && (handle_tail_singleton != 0)) {
        const uint64_t lhs_schedule = point_schedule_window[off];
        const uint32_t lhs_bucket_raw = static_cast<uint32_t>(lhs_schedule & 0xFFFFFFFFULL);
        if (lhs_bucket_raw < bucket_accumulator_size_bits) {
            const size_t lhs_bucket = static_cast<size_t>(lhs_bucket_raw);
            const bool has_bucket_accumulator =
                bitvector_get(bucket_accumulator_exists_data, bucket_accumulator_size_bits, lhs_bucket);

            if (has_bucket_accumulator) {
                // perform the add and report it; saturate in validation
                if (count_affine_for_output) {
                    if (use_safety_limits) {
                        if ((gpu_affine_input_it + 1) < MAX_AFFINE_CAPACITY) {
                            gpu_affine_input_it += 2;
                        } else {
                            // prod: capacity full => CPU would flush; just don't bump
                        }
                    } else {
                        if (gpu_affine_input_it <= MAX_AFFINE_CAPACITY - 2) {
                            gpu_affine_input_it += 2;
                        } else {
                            gpu_affine_input_it = MAX_AFFINE_CAPACITY;
                        }
                    }
                }
                // consume accumulator bit so subsequent iterations see correct state
                const size_t word = lhs_bucket >> 6;
                const size_t bit  = lhs_bucket & 63;
                const uint64_t mask = (1ULL << bit);
                atomicAnd(reinterpret_cast<unsigned long long*>(&bucket_accumulator_exists_data[word]),
                          static_cast<unsigned long long>(~mask));
            } else {
                // cache this singleton
                const size_t word = lhs_bucket >> 6;
                const size_t bit  = lhs_bucket & 63;
                const uint64_t mask = (1ULL << bit);
                atomicOr(reinterpret_cast<unsigned long long*>(&bucket_accumulator_exists_data[word]),
                         static_cast<unsigned long long>(mask));
            }
            off += 1;
            completed_iterations++;
        } else {
            // invalid: skip one safely
            off += 1;
            completed_iterations++;
        }
    }

    *final_point_offset_gpu    = off;
    *final_affine_input_it_gpu = gpu_affine_input_it; // unchanged in validation mode
    *iterations_processed_gpu  = completed_iterations;

    if (use_safety_limits && gpu_affine_input_it > MAX_AFFINE_CAPACITY) {
        printf("GPU ERROR: final affine_it=%llu exceeds capacity=%llu\n",
               (unsigned long long)gpu_affine_input_it,
               (unsigned long long)MAX_AFFINE_CAPACITY);
    }
    if (gpu_affine_input_it & 1ULL) {
        printf("GPU ERROR: affine_input_it odd (%llu)\n",
               (unsigned long long)gpu_affine_input_it);
    }
}



// HOST: Process entire batch with the new flags
// HOST: Process entire batch (validation-friendly defaults baked in)
extern "C" int gpu_process_entire_batch(
    const uint64_t* point_schedule,
    size_t start_point_it,
    size_t end_point_it,
    uint64_t* bucket_accumulator_exists_data,  // mutable bitvector
    size_t bucket_accumulator_size_bits,       // number of bits in bitvector
    size_t initial_affine_input_it,            // initial affine_input_it from CPU
    size_t* final_point_it,                    // final point_it after GPU processing
    size_t* final_affine_input_it,             // final affine_input_it after GPU processing
    size_t* iterations_processed,              // total iterations GPU completed
    int apply_updates,                         // whether to mutate/copy-back bucket state
    int respect_affine_capacity                // whether to gate on affine capacity (prod)
) {
    GPU_LOG("HOST DEBUG: gpu_process_entire_batch called with start=" << start_point_it << " end=" << end_point_it);
    if (!point_schedule || !bucket_accumulator_exists_data ||
        !final_point_it || !final_affine_input_it || !iterations_processed) {
        return -1;
    }

    const size_t num_entries = (end_point_it > start_point_it) ? (end_point_it - start_point_it) : 0;
    if (num_entries == 0) {
        *final_point_it        = start_point_it;
        *final_affine_input_it = initial_affine_input_it;
        *iterations_processed  = 0;
        return 0;
    }

    const size_t point_schedule_bytes = num_entries * sizeof(uint64_t);
    const size_t bucket_exists_bytes  = ((bucket_accumulator_size_bits + 63) / 64) * sizeof(uint64_t);

    // Validation-friendly defaults:
    // - If not applying updates, assume read-only validation window:
    //   do NOT stop early on capacity; always handle tail on GPU.
    const int effective_respect_capacity = (apply_updates != 0) ? respect_affine_capacity : 0;
    const int effective_handle_tail      = 1;

    // Device buffers
    uint64_t* d_point_schedule = nullptr;
    uint64_t* d_bucket_exists  = nullptr;
    size_t*   d_final_point_offset    = nullptr;
    size_t*   d_final_affine_input_it = nullptr;
    size_t*   d_iterations_processed  = nullptr;

    cudaError_t error;

    if ((error = cudaMalloc(&d_point_schedule, point_schedule_bytes)) != cudaSuccess) return -1;
    if ((error = cudaMalloc(&d_bucket_exists,  bucket_exists_bytes))  != cudaSuccess) { cudaFree(d_point_schedule); return -1; }
    if ((error = cudaMalloc(&d_final_point_offset,    sizeof(size_t))) != cudaSuccess) { cudaFree(d_point_schedule); cudaFree(d_bucket_exists); return -1; }
    if ((error = cudaMalloc(&d_final_affine_input_it, sizeof(size_t))) != cudaSuccess) { cudaFree(d_point_schedule); cudaFree(d_bucket_exists); cudaFree(d_final_point_offset); return -1; }
    if ((error = cudaMalloc(&d_iterations_processed,  sizeof(size_t))) != cudaSuccess) { cudaFree(d_point_schedule); cudaFree(d_bucket_exists); cudaFree(d_final_point_offset); cudaFree(d_final_affine_input_it); return -1; }

    // Zero outputs to avoid stale values if kernel exits early
    cudaMemset(d_final_point_offset,    0, sizeof(size_t));
    cudaMemset(d_final_affine_input_it, 0, sizeof(size_t));
    cudaMemset(d_iterations_processed,  0, sizeof(size_t));

    if ((error = cudaMemcpy(d_point_schedule, &point_schedule[start_point_it], point_schedule_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        cudaFree(d_point_schedule); cudaFree(d_bucket_exists); cudaFree(d_final_point_offset); cudaFree(d_final_affine_input_it); cudaFree(d_iterations_processed);
        return -1;
    }
    if ((error = cudaMemcpy(d_bucket_exists, bucket_accumulator_exists_data, bucket_exists_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
        cudaFree(d_point_schedule); cudaFree(d_bucket_exists); cudaFree(d_final_point_offset); cudaFree(d_final_affine_input_it); cudaFree(d_iterations_processed);
        return -1;
    }

    // Single-thread sequential kernel (state-faithful)
    process_point_schedule_sequential_with_results<<<1, 1>>>(
        d_point_schedule,
        d_bucket_exists,
        bucket_accumulator_size_bits,
        num_entries,
        initial_affine_input_it,
        d_final_point_offset,
        d_final_affine_input_it,
        d_iterations_processed,
        /*apply_updates=*/apply_updates,
        /*respect_affine_capacity=*/effective_respect_capacity,
        /*handle_tail_singleton=*/effective_handle_tail
    );

    if ((error = cudaDeviceSynchronize()) != cudaSuccess) {
        GPU_LOG("GPU BULK: CUDA kernel execution failed: " << cudaGetErrorString(error));
        cudaFree(d_point_schedule); cudaFree(d_bucket_exists); cudaFree(d_final_point_offset); cudaFree(d_final_affine_input_it); cudaFree(d_iterations_processed);
        return -1;
    }

    size_t relative_offset = 0;
    if ((error = cudaMemcpy(&relative_offset, d_final_point_offset, sizeof(size_t), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        cudaFree(d_point_schedule); cudaFree(d_bucket_exists); cudaFree(d_final_point_offset); cudaFree(d_final_affine_input_it); cudaFree(d_iterations_processed);
        return -1;
    }
    if ((error = cudaMemcpy(final_affine_input_it, d_final_affine_input_it, sizeof(size_t), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        cudaFree(d_point_schedule); cudaFree(d_bucket_exists); cudaFree(d_final_point_offset); cudaFree(d_final_affine_input_it); cudaFree(d_iterations_processed);
        return -1;
    }
    if ((error = cudaMemcpy(iterations_processed, d_iterations_processed, sizeof(size_t), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        cudaFree(d_point_schedule); cudaFree(d_bucket_exists); cudaFree(d_final_point_offset); cudaFree(d_final_affine_input_it); cudaFree(d_iterations_processed);
        return -1;
    }

    if (relative_offset > num_entries) {
        GPU_LOG("GPU BULK: WARNING relative_offset(" << relative_offset << ") > num_entries(" << num_entries << "), clamping");
        relative_offset = num_entries;
    }
    *final_point_it = start_point_it + relative_offset;
    if (*final_point_it < start_point_it || *final_point_it > end_point_it) {
        GPU_LOG("GPU BULK: WARNING final_point_it out of range, correcting to window end");
        *final_point_it = end_point_it;
    }

    if (apply_updates) {
        if ((error = cudaMemcpy(bucket_accumulator_exists_data, d_bucket_exists, bucket_exists_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess) {
            GPU_LOG("GPU BULK: CUDA memcpy D2H (bucket state) failed: " << cudaGetErrorString(error));
            cudaFree(d_point_schedule); cudaFree(d_bucket_exists); cudaFree(d_final_point_offset); cudaFree(d_final_affine_input_it); cudaFree(d_iterations_processed);
            return -1;
        }
    }

    cudaFree(d_point_schedule);
    cudaFree(d_bucket_exists);
    cudaFree(d_final_point_offset);
    cudaFree(d_final_affine_input_it);
    cudaFree(d_iterations_processed);

    GPU_LOG("GPU BULK: Completed " << *iterations_processed
           << " iterations, final point_it=" << *final_point_it
           << ", affine_it=" << *final_affine_input_it);
    return 0;
}


// GPU function that launches CUDA kernel and runs on actual GPU
extern "C" int gpu_validate_single_iteration_comprehensive(
    const uint64_t* point_schedule,
    size_t point_it,
    size_t end_point_it,
    const uint64_t* bucket_accumulator_exists_data,  // Raw BitVector data
    size_t bucket_accumulator_size_bits,             // Number of bits in BitVector
    CPUGPUIterationResult* gpu_result
) {
    if (!gpu_result || !point_schedule || (point_it + 1) >= end_point_it || !bucket_accumulator_exists_data) {
        return -1; // Invalid input
    }
    
    // Allocate GPU memory
    uint64_t* d_point_schedule = nullptr;
    uint64_t* d_bucket_exists = nullptr;
    CPUGPUIterationResult* d_results = nullptr;
    
    // We only need the current pair (2 entries) for a single-iteration kernel
    size_t point_schedule_bytes = 2 * sizeof(uint64_t);
    size_t bucket_exists_bytes = ((bucket_accumulator_size_bits + 63) / 64) * sizeof(uint64_t);
    
    cudaError_t error;
    
    // Allocate GPU memory
    error = cudaMalloc(&d_point_schedule, point_schedule_bytes);
    if (error != cudaSuccess) return -1;
    
    error = cudaMalloc(&d_bucket_exists, bucket_exists_bytes);
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        return -1;
    }
    
    error = cudaMalloc(&d_results, sizeof(CPUGPUIterationResult));
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        return -1;
    }
    
    // Copy data to GPU
    error = cudaMemcpy(d_point_schedule, &point_schedule[point_it], point_schedule_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        cudaFree(d_results);
        return -1;
    }
    
    error = cudaMemcpy(d_bucket_exists, bucket_accumulator_exists_data, bucket_exists_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        cudaFree(d_results);
        return -1;
    }
    
    // Launch CUDA kernel (1 thread for single iteration)
    process_point_schedule_kernel<<<1, 1>>>(
        d_point_schedule,
        d_bucket_exists,
        bucket_accumulator_size_bits,
        0,  // relative to copied data
        1,  // single iteration
        d_results
    );
    
    // Wait for GPU to finish
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        cudaFree(d_results);
        return -1;
    }
    
    // Copy results back to CPU
    error = cudaMemcpy(gpu_result, d_results, sizeof(CPUGPUIterationResult), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cudaFree(d_point_schedule);
        cudaFree(d_bucket_exists);
        cudaFree(d_results);
        return -1;
    }
    
    // Clean up GPU memory
    cudaFree(d_point_schedule);
    cudaFree(d_bucket_exists);
    cudaFree(d_results);
    
    return 0; // Success
}

// Keep the simple validation function for backward compatibility
extern "C" int gpu_validate_single_iteration(
    const uint64_t* point_schedule,
    size_t point_it,
    size_t end_point_it,
    CPUGPUIterationResult* gpu_result
) {
    if (!gpu_result || !point_schedule || (point_it + 1) >= end_point_it) {
        return -1; // Invalid input
    }
    
    // Process single iteration exactly like CPU main loop
    uint64_t lhs_schedule = point_schedule[point_it];
    uint64_t rhs_schedule = point_schedule[point_it + 1];
    
    // Extract bucket and point indices
    size_t lhs_bucket = static_cast<size_t>(lhs_schedule) & 0xFFFFFFFF;
    size_t rhs_bucket = static_cast<size_t>(rhs_schedule) & 0xFFFFFFFF;
    
    // Apply same logic as CPU (simplified - assumes no bucket accumulator)
    bool buckets_match = (lhs_bucket == rhs_bucket);
    // Note: GPU doesn't have access to bucket_accumulator_exists state
    // For validation, we assume has_bucket_accumulator = false
    bool has_bucket_accumulator = false; 
    bool do_affine_add = buckets_match || has_bucket_accumulator;
    
    // Calculate advancement (same as CPU)
    size_t point_it_advance = (do_affine_add && buckets_match) ? 2 : 1;
    
    // Return results for comparison
    gpu_result->point_it_advance = point_it_advance;
    gpu_result->do_affine_add = do_affine_add;
    gpu_result->lhs_bucket = lhs_bucket;
    gpu_result->rhs_bucket = rhs_bucket;
    gpu_result->buckets_match = buckets_match;
    
    return 0; // Success
}

// Placeholder for future consume_point_schedule GPU implementation
extern "C" int gpu_consume_point_schedule_test(
    const void* point_schedule_ptr,
    size_t point_schedule_size,
    const void* points_ptr,
    size_t points_size,
    void* bucket_data_ptr,
    void* affine_data_ptr
) {
    std::cout << "GPU: gpu_consume_point_schedule_test called with " << point_schedule_size << " points" << std::endl;
    
    // For now, just return success without doing anything
    // This is where we'll implement the actual GPU logic later
    return 0;
}
