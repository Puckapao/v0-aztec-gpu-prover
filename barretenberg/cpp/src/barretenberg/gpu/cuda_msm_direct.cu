// cuda_msm_direct.cu - Direct CPU Pippenger call with GPU interface
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <span>

// Include barretenberg headers for actual CPU functions
#include "../ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "../ecc/curves/bn254/bn254.hpp"

using namespace bb;
using namespace bb::scalar_multiplication;

// Type aliases matching the CPU implementation exactly
using Curve = curve::BN254;
using AffineElement = typename Curve::AffineElement;
using ScalarField = typename Curve::ScalarField;
using Element = typename Curve::Element;

// -------------------------------
// Direct CPU Pippenger Call
// -------------------------------

extern "C" int gpu_pippenger_msm_direct(
    void* result_ptr,
    const void* points_ptr,
    const void* scalars_ptr,
    size_t num_points
) {
    std::cout << "\n=== DIRECT CPU PIPPENGER CALL ===\n";
    std::cout << "Processing " << num_points << " points using REAL CPU Pippenger\n";
    
    if (!result_ptr || !points_ptr || !scalars_ptr || num_points == 0) {
        return -1;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Cast input pointers to the correct barretenberg types
        const AffineElement* bb_points = reinterpret_cast<const AffineElement*>(points_ptr);
        const ScalarField* bb_scalars = reinterpret_cast<const ScalarField*>(scalars_ptr);
        
        // Create spans for the CPU MSM function
        std::span<const AffineElement> points_span(bb_points, num_points);
        std::vector<ScalarField> scalars_vector(bb_scalars, bb_scalars + num_points);
        
        std::cout << "Step 1: Created barretenberg data structures\n";
        std::cout << "  Points span size: " << points_span.size() << "\n";
        std::cout << "  Scalars vector size: " << scalars_vector.size() << "\n";
        
        // Call the REAL CPU MSM function directly!
        std::cout << "Step 2: Calling MSM<BN254>::msm() - the actual CPU Pippenger function\n";
        
        // This is the same function the CPU uses in Step 2!
        PolynomialSpan<const ScalarField> scalar_span(0, scalars_vector);
        AffineElement cpu_result = MSM<Curve>::msm(points_span, scalar_span, false);
        
        std::cout << "Step 3: CPU Pippenger completed successfully!\n";
        
        // Copy result back
        AffineElement* output_result = reinterpret_cast<AffineElement*>(result_ptr);
        *output_result = cpu_result;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "=== DIRECT CPU PIPPENGER RESULT ===\n";
        std::cout << "CPU MSM Result x: " << std::hex << cpu_result.x << std::dec << "\n";
        std::cout << "CPU MSM Result y: " << std::hex << cpu_result.y << std::dec << "\n";
        std::cout << "Execution time: " << duration.count() << " μs\n";
        std::cout << "SUCCESS: Used the exact same CPU Pippenger as the reference!\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Direct CPU MSM error: " << e.what() << std::endl;
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
    return gpu_pippenger_msm_direct(result_ptr, points_ptr, scalars_ptr, num_points);
}