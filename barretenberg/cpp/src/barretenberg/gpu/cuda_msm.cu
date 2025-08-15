// cuda_msm.cu v0.0.7 - Fixed field inversion
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <span>

// -------------------------------
// Constants (BN254 Fq / Fr)
// -------------------------------

// Fq modulus p
__constant__ uint64_t BN254_FQ_MOD[4] = {
    0x3c208c16d87cfd47ULL,
    0x97816a916871ca8dULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};
// Fq R^2 mod p (little-endian limbs)
__constant__ uint64_t BN254_FQ_R2[4] = {
    0xf32cfc5b538afa89ULL,
    0xb5e71911d44501fbULL,
    0x47ab1eff0a417ff6ULL,
    0x06d89f71cab8351fULL
};
// Fq ninv = -p^{-1} mod 2^64
__constant__ uint64_t BN254_FQ_NINV = 0x87d20782e4866389ULL;

// Barretenberg G1 affine point at infinity (x = Fq::modulus, y=0)
__constant__ uint64_t POINT_AT_INFINITY_X[4] = {
    0x3C208C16D87CFD47ULL,
    0x97816a916871ca8dULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};
__constant__ uint64_t POINT_AT_INFINITY_Y[4] = {0,0,0,0};

// Host copies where needed
static const uint64_t BN254_FQ_MOD_HOST[4] = {
    0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t BN254_FQ_R2_HOST[4] = {
    0xf32cfc5b538afa89ULL, 0xb5e71911d44501fbULL,
    0x47ab1eff0a417ff6ULL, 0x06d89f71cab8351fULL
};
static const uint64_t BN254_FQ_NINV_HOST = 0x87d20782e4866389ULL;

static const uint64_t POINT_AT_INFINITY_X_HOST[4] = {
    0x3C208C16D87CFD47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t POINT_AT_INFINITY_Y_HOST[4] = {0,0,0,0};

// Fr (scalar field) modulus r, R^2, ninv — used on HOST to decode scalars
static const uint64_t BN254_FR_MOD_HOST[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t BN254_FR_R2_HOST[4] = {
    0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL, 0x216d0b17f4e44a5ULL
};
static const uint64_t BN254_FR_NINV_HOST = 0xc2e1f593efffffffULL; // -r^{-1} mod 2^64

// -------------------------------
// Data types
// -------------------------------
struct FieldElement { uint64_t limbs[4]; };
struct AffinePoint  { FieldElement x, y; };
struct JacobianPoint{ FieldElement X, Y, Z; };

struct SpanInfo {
    void*  data_ptr;  // pointer to array
    size_t size;      // number of elements
};

// -------------------------------
// Device helpers (Fq Montgomery)
// -------------------------------
__device__ __forceinline__ void add64_c(uint64_t a, uint64_t b, uint64_t& out, uint64_t& carry) {
    uint64_t old_carry = carry;
    uint64_t s = a + b;
    uint64_t c1 = (s < a) ? 1 : 0;
    uint64_t s2 = s + old_carry;
    uint64_t c2 = (s2 < s) ? 1 : 0;
    out = s2;
    carry = c1 + c2;
}
__device__ __forceinline__ void sub64_b(uint64_t a, uint64_t b, uint64_t& out, uint64_t& borrow) {
    uint64_t t = a - b - borrow;
    borrow = (a < (b + borrow));
    out = t;
}
__device__ __forceinline__ bool fe_is_zero(const FieldElement& a) {
    return (a.limbs[0]|a.limbs[1]|a.limbs[2]|a.limbs[3]) == 0ULL;
}
__device__ __forceinline__ bool fe_ge_mod(const FieldElement& a) {
    for (int i = 3; i >= 0; --i) {
        if (a.limbs[i] > BN254_FQ_MOD[i]) return true;
        if (a.limbs[i] < BN254_FQ_MOD[i]) return false;
    }
    return true; // equal
}
__device__ void fe_cond_sub_p(FieldElement& a) {
    if (fe_ge_mod(a)) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            uint64_t out;
            sub64_b(a.limbs[i], BN254_FQ_MOD[i], out, borrow);
            a.limbs[i] = out;
        }
    }
}
__device__ void fe_add(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) add64_c(a.limbs[i], b.limbs[i], r.limbs[i], carry);
    fe_cond_sub_p(r);
}
__device__ void fe_sub(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t out;
        sub64_b(a.limbs[i], b.limbs[i], out, borrow);
        r.limbs[i] = out;
    }
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 4; ++i) add64_c(r.limbs[i], BN254_FQ_MOD[i], r.limbs[i], carry);
    }
}
__device__ void mont_mul(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint64_t t[8] = {0};
    
    // Phase 1: Multiply a * b - exactly match host logic
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            // Compute a[i] * b[j] as 128-bit value
            uint64_t prod_lo = a.limbs[i] * b.limbs[j];
            uint64_t prod_hi = __umul64hi(a.limbs[i], b.limbs[j]);
            
            // Add to t[i+j] + carry
            uint64_t sum_lo = t[i+j] + prod_lo + carry;
            uint64_t sum_hi = prod_hi;
            
            // Handle overflow from low addition
            if (sum_lo < t[i+j] || (sum_lo == t[i+j] && (prod_lo + carry) != 0)) {
                sum_hi++;
            }
            
            t[i+j] = sum_lo;
            carry = sum_hi;
        }
        
        // Propagate remaining carry
        if (carry) {
            int k = i + 4;
            while (carry && k < 8) {
                t[k] += carry;
                carry = (t[k] < carry) ? 1 : 0;
                ++k;
            }
        }
    }
    
    // Phase 2: Montgomery reduction - exactly match host logic  
    for (int i = 0; i < 4; ++i) {
        uint64_t m = t[i] * BN254_FQ_NINV;
        uint64_t carry = 0;
        
        for (int j = 0; j < 4; ++j) {
            // Compute m * modulus[j] as 128-bit value
            uint64_t prod_lo = m * BN254_FQ_MOD[j];
            uint64_t prod_hi = __umul64hi(m, BN254_FQ_MOD[j]);
            
            // Add to t[i+j] + carry
            uint64_t sum_lo = t[i+j] + prod_lo + carry;
            uint64_t sum_hi = prod_hi;
            
            // Handle overflow from low addition
            if (sum_lo < t[i+j] || (sum_lo == t[i+j] && (prod_lo + carry) != 0)) {
                sum_hi++;
            }
            
            t[i+j] = sum_lo;
            carry = sum_hi;
        }
        
        // Propagate remaining carry
        if (carry) {
            int k = i + 4;
            while (carry && k < 8) {
                t[k] += carry;
                carry = (t[k] < carry) ? 1 : 0;
                ++k;
            }
        }
    }

    // Extract result from upper half
    r.limbs[0] = t[4];
    r.limbs[1] = t[5];
    r.limbs[2] = t[6];
    r.limbs[3] = t[7];
    
    // Final conditional subtraction if needed
    fe_cond_sub_p(r);
}
__device__ inline void fe_mul(FieldElement& r, const FieldElement& a, const FieldElement& b) { mont_mul(r,a,b); }
__device__ inline void fe_sqr(FieldElement& r, const FieldElement& a) { mont_mul(r,a,a); }
__device__ void fe_set_one(FieldElement& r) {
    FieldElement one = {1,0,0,0};
    FieldElement r2 = { BN254_FQ_R2[0], BN254_FQ_R2[1], BN254_FQ_R2[2], BN254_FQ_R2[3] };
    mont_mul(r, one, r2); // R
}
__device__ bool affine_is_infinity(const AffinePoint& P) {
    return (P.x.limbs[0]==POINT_AT_INFINITY_X[0] &&
            P.x.limbs[1]==POINT_AT_INFINITY_X[1] &&
            P.x.limbs[2]==POINT_AT_INFINITY_X[2] &&
            P.x.limbs[3]==POINT_AT_INFINITY_X[3] &&
            P.y.limbs[0]==0 && P.y.limbs[1]==0 && P.y.limbs[2]==0 && P.y.limbs[3]==0);
}

// -------------------------------
// Jacobian (device)
// -------------------------------
__device__ void jacobian_set_inf(JacobianPoint& R) { R.X={0,0,0,0}; R.Y={0,0,0,0}; R.Z={0,0,0,0}; }
__device__ bool jacobian_is_inf(const JacobianPoint& P) { return fe_is_zero(P.Z); }
__device__ void jacobian_from_affine(JacobianPoint& R, const AffinePoint& P) {
    if (affine_is_infinity(P)) { jacobian_set_inf(R); return; }
    R.X = P.x; R.Y = P.y; fe_set_one(R.Z);
}
__device__ void jacobian_double(JacobianPoint& R, const JacobianPoint& P) {
    if (jacobian_is_inf(P)) { R = P; return; }
    if (fe_is_zero(P.Y)) { jacobian_set_inf(R); return; }

    FieldElement A,B,C,D,E,F,tmp1,tmp2;
    fe_sqr(A, P.X);
    fe_sqr(B, P.Y);
    fe_sqr(C, B);

    fe_add(tmp1, P.X, B); fe_sqr(tmp1, tmp1);
    fe_sub(tmp1, tmp1, A); fe_sub(tmp1, tmp1, C);
    fe_add(D, tmp1, tmp1);

    fe_add(E, A, A); fe_add(E, E, A);
    fe_sqr(F, E);

    FieldElement twoD; fe_add(twoD, D, D);
    fe_sub(R.X, F, twoD);

    fe_sub(tmp1, D, R.X);
    fe_add(tmp2, C, C); fe_add(tmp2, tmp2, tmp2); fe_add(tmp2, tmp2, tmp2);
    fe_mul(tmp1, E, tmp1);
    fe_sub(R.Y, tmp1, tmp2);

    fe_mul(tmp1, P.Y, P.Z);
    fe_add(R.Z, tmp1, tmp1);
}

// FIX 1: correct mixed-add Z computation: Z3 = 2 * Z1 * H
__device__ void jacobian_add_mixed(JacobianPoint& R, const JacobianPoint& P, const AffinePoint& Q) {
    if (jacobian_is_inf(P)) { jacobian_from_affine(R, Q); return; }
    if (affine_is_infinity(Q)) { R = P; return; }

    FieldElement Z1Z1; fe_sqr(Z1Z1, P.Z);
    FieldElement U2;   fe_mul(U2, Z1Z1, Q.x);
    FieldElement Z1_cu; fe_mul(Z1_cu, Z1Z1, P.Z);
    FieldElement S2;   fe_mul(S2, Z1_cu, Q.y);

    FieldElement H; fe_sub(H, U2, P.X);
    FieldElement I; FieldElement twoH; fe_add(twoH,H,H); fe_sqr(I, twoH);
    FieldElement J; fe_mul(J, H, I);
    FieldElement S2_minus_S1; fe_sub(S2_minus_S1, S2, P.Y);
    if (fe_is_zero(H)) { if (fe_is_zero(S2_minus_S1)) { jacobian_double(R, P); } else { jacobian_set_inf(R);} return; }

    FieldElement r; fe_add(r, S2_minus_S1, S2_minus_S1);
    FieldElement V; fe_mul(V, P.X, I);
    FieldElement r2; fe_sqr(r2, r);
    FieldElement X3; FieldElement tmp; fe_sub(tmp, r2, J);
    FieldElement twoV; fe_add(twoV, V, V);
    fe_sub(X3, tmp, twoV);

    FieldElement Y3; FieldElement VminusX3; fe_sub(VminusX3, V, X3);
    FieldElement S1; fe_add(S1, P.Y, P.Y); // 2*S1
    FieldElement S1J2; FieldElement S1J; fe_mul(S1J, S1, J); fe_add(S1J2,S1J,S1J);
    FieldElement t; fe_mul(t, r, VminusX3);
    fe_sub(Y3, t, S1J2);

    // Z3 = 2 * Z1 * H   (correct formula for mixed add)
    FieldElement Z3; FieldElement twoZ1; fe_add(twoZ1, P.Z, P.Z);
    fe_mul(Z3, twoZ1, H);

    R.X = X3; R.Y = Y3; R.Z = Z3;
}
__device__ void jacobian_add(JacobianPoint& R, const JacobianPoint& P, const JacobianPoint& Q) {
    if (jacobian_is_inf(P)) { R = Q; return; }
    if (jacobian_is_inf(Q)) { R = P; return; }

    FieldElement Z1Z1; fe_sqr(Z1Z1, P.Z);
    FieldElement Z2Z2; fe_sqr(Z2Z2, Q.Z);

    FieldElement U1; fe_mul(U1, P.X, Z2Z2);
    FieldElement U2; fe_mul(U2, Q.X, Z1Z1);

    FieldElement Z2_cu; fe_mul(Z2_cu, Z2Z2, Q.Z);
    FieldElement Z1_cu; fe_mul(Z1_cu, Z1Z1, P.Z);

    FieldElement S1; fe_mul(S1, P.Y, Z2_cu);
    FieldElement S2; fe_mul(S2, Q.Y, Z1_cu);

    FieldElement H; fe_sub(H, U2, U1);
    FieldElement r; fe_sub(r, S2, S1); FieldElement two_r; fe_add(two_r, r, r);

    if (fe_is_zero(H)) {
        if (fe_is_zero(r)) { jacobian_double(R, P); return; }
        jacobian_set_inf(R); return;
    }

    FieldElement I; FieldElement twoH; fe_add(twoH,H,H); fe_sqr(I, twoH);
    FieldElement J; fe_mul(J, H, I);
    FieldElement V; fe_mul(V, U1, I);
    FieldElement r2; fe_sqr(r2, two_r);

    FieldElement X3; FieldElement tmp; fe_sub(tmp, r2, J);
    FieldElement twoV; fe_add(twoV, V, V);
    fe_sub(X3, tmp, twoV);

    FieldElement Y3; FieldElement VminusX3; fe_sub(VminusX3, V, X3);
    FieldElement S1J2; FieldElement S1J; fe_mul(S1J, S1, J); fe_add(S1J2, S1J, S1J);
    FieldElement t; fe_mul(t, two_r, VminusX3);
    fe_sub(Y3, t, S1J2);

    FieldElement Z3; FieldElement Z1plusZ2; fe_add(Z1plusZ2, P.Z, Q.Z);
    fe_sqr(Z3, Z1plusZ2);
    fe_sub(Z3, Z3, Z1Z1);
    fe_sub(Z3, Z3, Z2Z2);
    fe_mul(Z3, Z3, H);

    R.X = X3; R.Y = Y3; R.Z = Z3;
}

// -------------------------------
// Kernel: per-point scalar mul (w=4) → Jacobian
// -------------------------------
// Simple scalar multiplication kernel matching naive_msm approach
__global__ void simple_scalar_mul_kernel(
    JacobianPoint* __restrict__ d_out,
    const AffinePoint* __restrict__ d_points,
    const FieldElement* __restrict__ d_scalars,
    size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t)idx >= n) return;

    const AffinePoint& P = d_points[idx];
    const FieldElement& k = d_scalars[idx];

    // Handle zero scalar or point at infinity
    if ((k.limbs[0]|k.limbs[1]|k.limbs[2]|k.limbs[3]) == 0ULL || affine_is_infinity(P)) {
        jacobian_set_inf(d_out[idx]);
        return;
    }

    // Convert affine point to Jacobian for accumulator
    JacobianPoint accumulator;
    jacobian_from_affine(accumulator, P);
    
    // Find MSB of scalar (same as CPU algorithm)
    int maximum_set_bit = -1;
    for (int i = 253; i >= 0; i--) {
        int limb = i / 64;
        int bit_pos = i % 64;
        if (limb < 4 && (k.limbs[limb] & (1ULL << bit_pos))) {
            maximum_set_bit = i;
            break;
        }
    }
    
    if (maximum_set_bit <= 0) {
        // Scalar is 0 or 1
        if (maximum_set_bit == 0) {
            // Scalar is 1, result = P
            accumulator = accumulator; // Already set to P
        } else {
            // Scalar is 0, result = infinity
            jacobian_set_inf(accumulator);
        }
        d_out[idx] = accumulator;
        return;
    }
    
    // Exact CPU algorithm: start from MSB-1 and work down
    for (int i = maximum_set_bit - 1; i >= 0; i--) {
        jacobian_double(accumulator, accumulator); // accumulator.self_dbl()
        
        // Check if bit i is set
        int limb = i / 64;
        int bit_pos = i % 64;
        if (limb < 4 && (k.limbs[limb] & (1ULL << bit_pos))) {
            // accumulator += *this (add original point P)
            JacobianPoint P_jacobian;
            jacobian_from_affine(P_jacobian, P);
            jacobian_add(accumulator, accumulator, P_jacobian);
        }
    }
    
    JacobianPoint result = accumulator;
    
    d_out[idx] = result;
    
    // Debug: Test with simple scalar multiplication: 2 * P
    if (idx == 0) {
        printf("GPU Debug - Point 0 scalar multiplication:\n");
        printf("  Input scalar: %016llx_%016llx_%016llx_%016llx\n", 
               k.limbs[3], k.limbs[2], k.limbs[1], k.limbs[0]);
        printf("  Input point X: %016llx_%016llx_%016llx_%016llx\n",
               P.x.limbs[3], P.x.limbs[2], P.x.limbs[1], P.x.limbs[0]);
        printf("  Input point Y: %016llx_%016llx_%016llx_%016llx\n",
               P.y.limbs[3], P.y.limbs[2], P.y.limbs[1], P.y.limbs[0]);
        
        // Test simple doubling: 2*P
        JacobianPoint test_double;
        jacobian_from_affine(test_double, P);
        jacobian_double(test_double, test_double);
        printf("  2*P Result X: %016llx_%016llx_%016llx_%016llx\n",
               test_double.X.limbs[3], test_double.X.limbs[2], test_double.X.limbs[1], test_double.X.limbs[0]);
        printf("  2*P Result Z: %016llx_%016llx_%016llx_%016llx\n",
               test_double.Z.limbs[3], test_double.Z.limbs[2], test_double.Z.limbs[1], test_double.Z.limbs[0]);
        
        printf("  Final Result X: %016llx_%016llx_%016llx_%016llx\n",
               result.X.limbs[3], result.X.limbs[2], result.X.limbs[1], result.X.limbs[0]);
        printf("  Final Result Y: %016llx_%016llx_%016llx_%016llx\n",
               result.Y.limbs[3], result.Y.limbs[2], result.Y.limbs[1], result.Y.limbs[0]);
        printf("  Final Result Z: %016llx_%016llx_%016llx_%016llx\n",
               result.Z.limbs[3], result.Z.limbs[2], result.Z.limbs[1], result.Z.limbs[0]);
    }
}

__global__ void msm_kernel(
    JacobianPoint* __restrict__ d_out,
    const AffinePoint* __restrict__ d_points,   // Fq Montgomery
    const FieldElement* __restrict__ d_scalars, // Fr STANDARD (little-endian limbs)
    size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((size_t)idx >= n) return;

    const AffinePoint& P = d_points[idx];
    const FieldElement& s = d_scalars[idx];

    if ((s.limbs[0]|s.limbs[1]|s.limbs[2]|s.limbs[3]) == 0ULL || affine_is_infinity(P)) {
        jacobian_set_inf(d_out[idx]);
        return;
    }

    // Precompute [1..15]P in Jacobian (via mixed adds)
    const int W = 4;
    const int T = 1 << W; // 16
    JacobianPoint table[T];
    jacobian_from_affine(table[1], P); // 1P
    for (int i = 2; i < T; ++i) jacobian_add_mixed(table[i], table[i-1], P);
    jacobian_set_inf(table[0]);

    // Windowed method (MSB→LSB) - process scalar from high to low bits
    JacobianPoint R; jacobian_set_inf(R);

    // Start from the most significant window and work down
    for (int win = 63; win >= 0; --win) {
        // Double the accumulator 4 times for the 4-bit window
        if (!jacobian_is_inf(R)) {
            for (int d = 0; d < W; ++d) {
                jacobian_double(R, R);
            }
        }
        
        // Extract 4-bit window from scalar
        int bit_pos = win * W; // Starting bit position for this window
        int limb_idx = bit_pos >> 6; // Which 64-bit limb
        int bit_offset = bit_pos & 63; // Bit offset within limb

        uint64_t window_val = 0;
        if (limb_idx < 4) {
            if (bit_offset <= 60) {
                // Window fits entirely in current limb
                window_val = (s.limbs[limb_idx] >> bit_offset) & 0xF;
            } else {
                // Window spans two limbs
                int bits_from_current = 64 - bit_offset;
                int bits_from_next = W - bits_from_current;
                
                window_val = (s.limbs[limb_idx] >> bit_offset) & ((1ULL << bits_from_current) - 1);
                if (limb_idx + 1 < 4) {
                    window_val |= (s.limbs[limb_idx + 1] & ((1ULL << bits_from_next) - 1)) << bits_from_current;
                }
            }
        }
        
        // Add the precomputed point if window is non-zero
        if (window_val > 0) {
            if (jacobian_is_inf(R)) {
                R = table[window_val];
            } else {
                jacobian_add(R, R, table[window_val]);
            }
        }
    }
    d_out[idx] = R;
}

// -------------------------------
// Host Fq Montgomery (for final reduction)
// -------------------------------
static inline void add64_c_h(uint64_t a, uint64_t b, uint64_t& out, uint64_t& carry) {
    __uint128_t s = ( (__uint128_t)a + b + carry );
    out = (uint64_t)s;
    carry = (uint64_t)(s >> 64);
}
static inline void sub64_b_h(uint64_t a, uint64_t b, uint64_t& out, uint64_t& borrow) {
    __uint128_t aa = a;
    __uint128_t bb = b + borrow;
    __uint128_t d = aa - bb;
    out = (uint64_t)d;
    borrow = (aa < bb) ? 1 : 0;
}
static inline void fq_cond_sub(FieldElement& a) {
    for (int i = 3; i >= 0; --i) {
        if (a.limbs[i] > BN254_FQ_MOD_HOST[i]) goto do_sub;
        if (a.limbs[i] < BN254_FQ_MOD_HOST[i]) return;
    }
do_sub:
    {
        uint64_t borrow=0;
        for (int i=0;i<4;++i) {
            uint64_t o;
            sub64_b_h(a.limbs[i], BN254_FQ_MOD_HOST[i], o, borrow);
            a.limbs[i] = o;
        }
    }
}
static inline void fq_add(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint64_t c=0; for (int i=0;i<4;++i) add64_c_h(a.limbs[i], b.limbs[i], r.limbs[i], c);
    fq_cond_sub(r);
}
static inline void fq_sub(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint64_t borrow=0; for (int i=0;i<4;++i) sub64_b_h(a.limbs[i], b.limbs[i], r.limbs[i], borrow);
    if (borrow) { uint64_t c=0; for (int i=0;i<4;++i) add64_c_h(r.limbs[i], BN254_FQ_MOD_HOST[i], r.limbs[i], c); }
}
static inline void fq_mont_mul(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint64_t t[8]={0};
    for (int i=0;i<4;++i) {
        __uint128_t carry=0;
        for (int j=0;j<4;++j) {
            __uint128_t acc = (__uint128_t)t[i+j] + (__uint128_t)a.limbs[i]*b.limbs[j] + (uint64_t)carry;
            t[i+j] = (uint64_t)acc;
            carry = (acc >> 64) + (uint64_t)(((__uint128_t)a.limbs[i]*b.limbs[j]) >> 64);
        }
        int k=i+4;
        while (carry && k<8) {
            __uint128_t acc = (__uint128_t)t[k] + (uint64_t)carry;
            t[k] = (uint64_t)acc;
            carry = (acc >> 64);
            ++k;
        }
    }
    for (int i=0;i<4;++i) {
        uint64_t m = (uint64_t)((__uint128_t)t[i] * BN254_FQ_NINV_HOST);
        __uint128_t carry=0;
        for (int j=0;j<4;++j) {
            __uint128_t acc = (__uint128_t)t[i+j] + (__uint128_t)m*BN254_FQ_MOD_HOST[j] + (uint64_t)carry;
            t[i+j] = (uint64_t)acc;
            carry = (acc >> 64);
        }
        int k=i+4;
        while (carry && k<8) {
            __uint128_t acc = (__uint128_t)t[k] + (uint64_t)carry;
            t[k] = (uint64_t)acc;
            carry = (acc >> 64);
            ++k;
        }
    }
    r.limbs[0]=t[4]; r.limbs[1]=t[5]; r.limbs[2]=t[6]; r.limbs[3]=t[7];
    fq_cond_sub(r);
}
static inline void fq_mul(FieldElement& r, const FieldElement& a, const FieldElement& b){ fq_mont_mul(r,a,b); }
static inline void fq_sqr(FieldElement& r, const FieldElement& a){ fq_mont_mul(r,a,a); }
static inline bool fq_is_zero(const FieldElement& a) {
    return (a.limbs[0]|a.limbs[1]|a.limbs[2]|a.limbs[3])==0ULL;
}
static inline void fq_set_one(FieldElement& r) {
    FieldElement one = {1,0,0,0};
    FieldElement r2 = { BN254_FQ_R2_HOST[0], BN254_FQ_R2_HOST[1], BN254_FQ_R2_HOST[2], BN254_FQ_R2_HOST[3] };
    fq_mont_mul(r, one, r2);
}

// Host Jacobian ops (for accumulation)
static inline void j_set_inf(JacobianPoint& R) { memset(&R, 0, sizeof(R)); }
static inline bool j_is_inf(const JacobianPoint& P) { return fq_is_zero(P.Z); }
static inline void j_double(JacobianPoint& R, const JacobianPoint& P) {
    if (j_is_inf(P) || fq_is_zero(P.Y)) { j_set_inf(R); return; }
    FieldElement A,B,C,D,E,F,tmp1,tmp2;

    fq_sqr(A, P.X);
    fq_sqr(B, P.Y);
    fq_sqr(C, B);

    fq_add(tmp1, P.X, B); fq_sqr(tmp1, tmp1);
    fq_sub(tmp1, tmp1, A); fq_sub(tmp1, tmp1, C);
    fq_add(D, tmp1, tmp1);

    fq_add(E, A, A); fq_add(E, E, A);
    fq_sqr(F, E);

    FieldElement twoD; fq_add(twoD,D,D);
    fq_sub(R.X, F, twoD);

    fq_sub(tmp1, D, R.X);
    fq_add(tmp2, C, C); fq_add(tmp2, tmp2, tmp2); fq_add(tmp2, tmp2, tmp2);
    fq_mul(tmp1, E, tmp1);
    fq_sub(R.Y, tmp1, tmp2);

    fq_mul(tmp1, P.Y, P.Z);
    fq_add(R.Z, tmp1, tmp1);
}
static inline void j_add(JacobianPoint& R, const JacobianPoint& P, const JacobianPoint& Q) {
    if (j_is_inf(P)) { R = Q; return; }
    if (j_is_inf(Q)) { R = P; return; }

    FieldElement Z1Z1; fq_sqr(Z1Z1, P.Z);
    FieldElement Z2Z2; fq_sqr(Z2Z2, Q.Z);
    FieldElement U1; fq_mul(U1, P.X, Z2Z2);
    FieldElement U2; fq_mul(U2, Q.X, Z1Z1);
    FieldElement Z1_cu; fq_mul(Z1_cu, Z1Z1, P.Z);
    FieldElement Z2_cu; fq_mul(Z2_cu, Z2Z2, Q.Z);
    FieldElement S1; fq_mul(S1, P.Y, Z2_cu);
    FieldElement S2; fq_mul(S2, Q.Y, Z1_cu);

    FieldElement H; fq_sub(H, U2, U1);
    FieldElement r; fq_sub(r, S2, S1); FieldElement two_r; fq_add(two_r, r, r);

    if (fq_is_zero(H)) {
        if (fq_is_zero(r)) { j_double(R, P); return; }
        j_set_inf(R); return;
    }

    FieldElement I; FieldElement twoH; fq_add(twoH,H,H); fq_sqr(I, twoH);
    FieldElement J; fq_mul(J, H, I);
    FieldElement V; fq_mul(V, U1, I);
    FieldElement r2; fq_sqr(r2, two_r);

    FieldElement X3; FieldElement tmp; fq_sub(tmp, r2, J);
    FieldElement twoV; fq_add(twoV, V, V);
    fq_sub(X3, tmp, twoV);

    FieldElement Y3; FieldElement VminusX3; fq_sub(VminusX3, V, X3);
    FieldElement S1J2; FieldElement S1J; fq_mul(S1J, S1, J); fq_add(S1J2, S1J, S1J);
    FieldElement t; fq_mul(t, two_r, VminusX3);
    fq_sub(Y3, t, S1J2);

    FieldElement Z3; FieldElement Z1plusZ2; fq_add(Z1plusZ2, P.Z, Q.Z);
    fq_sqr(Z3, Z1plusZ2);
    fq_sub(Z3, Z3, Z1Z1);
    fq_sub(Z3, Z3, Z2Z2);
    fq_mul(Z3, Z3, H);

    R.X = X3; R.Y = Y3; R.Z = Z3;
}
static inline void j_from_affine(JacobianPoint& R, const AffinePoint& P) {
    bool is_inf = true;
    for (int i=0;i<4;++i) {
        if (P.x.limbs[i] != POINT_AT_INFINITY_X_HOST[i]) { is_inf=false; break; }
    }
    if (is_inf) { j_set_inf(R); return; }
    R.X = P.x; R.Y = P.y; fq_set_one(R.Z);
}
static inline void fq_inv(FieldElement& out, const FieldElement& z) {
    // Use Fermat's Little Theorem: z^(p-2) = z^(-1) mod p
    // where p = BN254_FQ_MOD = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
    // so p-2 = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45
    
    static const uint64_t p_minus_2[4] = {
        0x3c208c16d87cfd45ULL,  // mod[0] - 2
        0x97816a916871ca8dULL,  // mod[1] 
        0xb85045b68181585dULL,  // mod[2]
        0x30644e72e131a029ULL   // mod[3]
    };
    
    FieldElement base = z;
    FieldElement res; fq_set_one(res);
    
    // Binary exponentiation using p-2 as exponent
    for (int limb = 0; limb < 4; ++limb) {
        uint64_t exp_limb = p_minus_2[limb];
        for (int bit = 0; bit < 64; ++bit) {
            if (exp_limb & (1ULL << bit)) {
                fq_mul(res, res, base);
            }
            fq_sqr(base, base);
        }
    }
    out = res;
}
static inline void j_to_affine(AffinePoint& A, const JacobianPoint& P) {
    if (j_is_inf(P)) {
        for (int i=0;i<4;++i) { A.x.limbs[i]=POINT_AT_INFINITY_X_HOST[i]; A.y.limbs[i]=0; }
        return;
    }
    FieldElement z_inv; fq_inv(z_inv, P.Z);
    FieldElement z2; fq_sqr(z2, z_inv);
    FieldElement z3; fq_mul(z3, z2, z_inv);
    fq_mul(A.x, P.X, z2);
    fq_mul(A.y, P.Y, z3);
}

// -------------------------------
// Fr (host) Montgomery decode for scalars
// -------------------------------
static inline void fr_mont_mul(FieldElement& r, const FieldElement& a, const FieldElement& b) {
    uint64_t t[8]={0};
    for (int i=0;i<4;++i) {
        __uint128_t carry=0;
        for (int j=0;j<4;++j) {
            __uint128_t acc = (__uint128_t)t[i+j] + (__uint128_t)a.limbs[i]*b.limbs[j] + (uint64_t)carry;
            t[i+j] = (uint64_t)acc;
            carry = (acc >> 64) + (uint64_t)(((__uint128_t)a.limbs[i]*b.limbs[j]) >> 64);
        }
        int k=i+4;
        while (carry && k<8) {
            __uint128_t acc = (__uint128_t)t[k] + (uint64_t)carry;
            t[k] = (uint64_t)acc;
            carry = (acc >> 64);
            ++k;
        }
    }
    for (int i=0;i<4;++i) {
        uint64_t m = (uint64_t)((__uint128_t)t[i] * BN254_FR_NINV_HOST);
        __uint128_t carry=0;
        for (int j=0;j<4;++j) {
            __uint128_t acc = (__uint128_t)t[i+j] + (__uint128_t)m*BN254_FR_MOD_HOST[j] + (uint64_t)carry;
            t[i+j] = (uint64_t)acc;
            carry = (acc >> 64);
        }
        int k=i+4;
        while (carry && k<8) {
            __uint128_t acc = (__uint128_t)t[k] + (uint64_t)carry;
            t[k] = (uint64_t)acc;
            carry = (acc >> 64);
            ++k;
        }
    }
    r.limbs[0]=t[4]; r.limbs[1]=t[5]; r.limbs[2]=t[6]; r.limbs[3]=t[7];
    for (int i=3;i>=0;--i) {
        if (r.limbs[i] > BN254_FR_MOD_HOST[i]) goto subr;
        if (r.limbs[i] < BN254_FR_MOD_HOST[i]) return;
    }
subr:
    {
        uint64_t borrow=0;
        for (int i=0;i<4;++i) {
            uint64_t o; sub64_b_h(r.limbs[i], BN254_FR_MOD_HOST[i], o, borrow);
            r.limbs[i] = o;
        }
    }
}
static inline void fr_from_mont(FieldElement& out, const FieldElement& a) {
    FieldElement one = {1,0,0,0};
    fr_mont_mul(out, a, one); // REDC once: a*1*R^{-1} => standard
}

// -------------------------------
// Exposed C interface
// -------------------------------
extern "C" __attribute__((visibility("default")))
int cuda_msm_compute(
    void* result,                   // out (array of AffinePoint, one per span)
    const void* points_spans,       // array of std::span<const AffineElement>
    const void* scalars_spans,      // array of std::span<ScalarField>  
    size_t num_spans)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Version logging
    std::cout << "=== libcuda_msm.so v0.0.7 called with " << num_spans << " spans ===" << std::endl;
    
    // Debug: Log span sizes
    using AffineSpan = std::span<const AffinePoint>;
    using ScalarSpan = std::span<FieldElement>;
    
    const AffineSpan* point_spans_debug = static_cast<const AffineSpan*>(points_spans);
    const ScalarSpan* scalar_spans_debug = static_cast<const ScalarSpan*>(scalars_spans);
    
    for (size_t s = 0; s < num_spans; ++s) {
        std::cout << "Span " << s << ": " << point_spans_debug[s].size() << " points, " 
                  << scalar_spans_debug[s].size() << " scalars" << std::endl;
        
        // Debug first point and scalar for span 0
        if (s == 0 && point_spans_debug[s].size() > 0) {
            const AffinePoint& p0 = point_spans_debug[s][0];
            const FieldElement& s0 = scalar_spans_debug[s][0];
            
            std::cout << "First point X: " << std::hex 
                      << p0.x.limbs[3] << "_" << p0.x.limbs[2] << "_" 
                      << p0.x.limbs[1] << "_" << p0.x.limbs[0] << std::dec << std::endl;
            std::cout << "First scalar: " << std::hex 
                      << s0.limbs[3] << "_" << s0.limbs[2] << "_" 
                      << s0.limbs[1] << "_" << s0.limbs[0] << std::dec << std::endl;
        }
    }
    
    if (num_spans == 0) {
        // Return array of points at infinity
        std::vector<AffinePoint> outs(1);
        for (int i=0;i<4;++i) { 
            outs[0].x.limbs[i] = POINT_AT_INFINITY_X_HOST[i]; 
            outs[0].y.limbs[i] = 0; 
        }
        memcpy(result, outs.data(), sizeof(AffinePoint));
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::cout << "GPU MSM total time: " << us << " us\n";
        return 0;
    }

    // CRITICAL FIX: The CPU is actually passing std::span<const AffineElement> and std::span<ScalarField>
    // But these might be different sizes than our FieldElement/AffinePoint structs
    // Let's cast more carefully and check sizes
    
    // For now, let's assume they match and cast directly to our types
    using AffineSpan = std::span<const AffinePoint>;
    using ScalarSpan = std::span<FieldElement>;
    
    const AffineSpan* point_spans = static_cast<const AffineSpan*>(points_spans);
    const ScalarSpan* scalar_spans = static_cast<const ScalarSpan*>(scalars_spans);
    
    // Log struct sizes for verification
    std::cout << "Our AffinePoint size: " << sizeof(AffinePoint) << ", FieldElement size: " << sizeof(FieldElement) << std::endl;
    
    // Calculate total points across all spans
    size_t total_points = 0;
    std::vector<size_t> span_sizes(num_spans);
    for (size_t s = 0; s < num_spans; ++s) {
        span_sizes[s] = std::min(point_spans[s].size(), scalar_spans[s].size());
        total_points += span_sizes[s];
    }
    
    if (total_points == 0) {
        // Return array of points at infinity
        std::vector<AffinePoint> outs(num_spans);
        for (size_t s = 0; s < num_spans; ++s) {
            for (int i=0;i<4;++i) { 
                outs[s].x.limbs[i] = POINT_AT_INFINITY_X_HOST[i]; 
                outs[s].y.limbs[i] = 0; 
            }
        }
        memcpy(result, outs.data(), num_spans * sizeof(AffinePoint));
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::cout << "GPU MSM total time: " << us << " us\n";
        return 0;
    }

    // Flatten all spans into single arrays
    std::vector<AffinePoint> h_points(total_points);
    std::vector<FieldElement> h_scalars_std(total_points);
    
    size_t offset = 0;
    for (size_t s = 0; s < num_spans; ++s) {
        const size_t span_size = span_sizes[s];
        
        // Use points exactly as CPU provides them (no coordinate conversion)
        for (size_t i = 0; i < span_size; ++i) {
            h_points[offset + i] = point_spans[s][i];
        }
        
        // CPU already converts scalars to standard form, so use them directly
        for (size_t i = 0; i < span_size; ++i) {
            h_scalars_std[offset + i] = scalar_spans[s][i];
            
            // Debug: Log scalar values for first few scalars of first span
            if (s == 0 && i < 3) {
                const FieldElement& scalar = h_scalars_std[offset + i];
                std::cout << "Scalar " << i << " (already standard): " << std::hex
                          << scalar.limbs[3] << "_" << scalar.limbs[2] << "_" << scalar.limbs[1] << "_" << scalar.limbs[0] << std::dec << std::endl;
            }
        }
        
        offset += span_size;
    }

    // Device buffers - declare all variables before any goto targets
    AffinePoint*   d_points  = nullptr;
    FieldElement*  d_scalars = nullptr;
    JacobianPoint* d_jac     = nullptr;
    cudaError_t err = cudaSuccess;
    const int TPB = 256;
    const int blocks = (int)((total_points + TPB - 1) / TPB);
    std::vector<JacobianPoint> h_jac;
    std::vector<AffinePoint> outputs;
    
    // Initialize vectors
    h_jac.reserve(total_points);
    outputs.reserve(num_spans);

    // Allocate device memory
    err = cudaMalloc(&d_points,  total_points * sizeof(AffinePoint));
    if (err) { std::cerr<<"cudaMalloc points failed\n"; goto CLEANUP; }
    
    err = cudaMalloc(&d_scalars, total_points * sizeof(FieldElement));
    if (err) { std::cerr<<"cudaMalloc scalars failed\n"; goto CLEANUP; }
    
    err = cudaMalloc(&d_jac,     total_points * sizeof(JacobianPoint));
    if (err) { std::cerr<<"cudaMalloc jac failed\n"; goto CLEANUP; }

    // Copy data to device
    err = cudaMemcpy(d_points, h_points.data(), total_points*sizeof(AffinePoint), cudaMemcpyHostToDevice);
    if (err) { std::cerr<<"cpy points->dev failed\n"; goto CLEANUP; }
    
    err = cudaMemcpy(d_scalars, h_scalars_std.data(), total_points*sizeof(FieldElement), cudaMemcpyHostToDevice);
    if (err) { std::cerr<<"cpy scalars->dev failed\n"; goto CLEANUP; }

    // Launch kernel using simple scalar multiplication (matching CPU)
    simple_scalar_mul_kernel<<<blocks, TPB>>>(d_jac, d_points, d_scalars, total_points);
    
    err = cudaGetLastError();
    if (err) { std::cerr<<"kernel launch failed: "<<cudaGetErrorString(err)<<"\n"; goto CLEANUP; }
    
    err = cudaDeviceSynchronize();
    if (err) { std::cerr<<"kernel exec failed: "<<cudaGetErrorString(err)<<"\n"; goto CLEANUP; }

    // Copy results back
    h_jac.resize(total_points);
    err = cudaMemcpy(h_jac.data(), d_jac, total_points*sizeof(JacobianPoint), cudaMemcpyDeviceToHost);
    if (err) { std::cerr<<"cpy jac->host failed\n"; goto CLEANUP; }
    
    // Debug: Check first few kernel results
    std::cout << "First 3 kernel results (Jacobian):" << std::endl;
    for (size_t i = 0; i < std::min((size_t)3, total_points); ++i) {
        const JacobianPoint& jp = h_jac[i];
        bool is_inf = j_is_inf(jp);
        std::cout << "  Point " << i << " (inf=" << is_inf << "): X=" << std::hex 
                  << jp.X.limbs[3] << "_" << jp.X.limbs[2] << "_" << jp.X.limbs[1] << "_" << jp.X.limbs[0]
                  << " Z=" << jp.Z.limbs[3] << "_" << jp.Z.limbs[2] << "_" << jp.Z.limbs[1] << "_" << jp.Z.limbs[0] << std::dec << std::endl;
    }

    // Accumulate per span and convert to affine
    outputs.resize(num_spans);
    offset = 0;
    
    for (size_t s = 0; s < num_spans; ++s) {
        const size_t span_size = span_sizes[s];
        
        JacobianPoint acc; 
        j_set_inf(acc);
        
        for (size_t i = 0; i < span_size; ++i) {
            if (j_is_inf(acc)) {
                acc = h_jac[offset + i];
            } else {
                j_add(acc, acc, h_jac[offset + i]);
            }
        }
        
        // Debug: Check accumulator state
        bool is_acc_inf = j_is_inf(acc);
        std::cout << "Span " << s << " accumulator: inf=" << is_acc_inf;
        if (is_acc_inf) {
            std::cout << " (returning sentinel)" << std::endl;
        } else {
            std::cout << " X=" << std::hex << acc.X.limbs[3] << "_" << acc.X.limbs[2] << "_" << acc.X.limbs[1] << "_" << acc.X.limbs[0] << std::dec << std::endl;
        }
        
        // CRITICAL FIX: Check if accumulator is still at infinity
        if (j_is_inf(acc)) {
            // Return the correct point at infinity that matches CPU expectations
            for (int i = 0; i < 4; ++i) { 
                outputs[s].x.limbs[i] = POINT_AT_INFINITY_X_HOST[i]; 
                outputs[s].y.limbs[i] = POINT_AT_INFINITY_Y_HOST[i]; 
            }
        } else {
            j_to_affine(outputs[s], acc);
        }
        
        offset += span_size;
    }
    
    memcpy(result, outputs.data(), num_spans * sizeof(AffinePoint));

CLEANUP:
    if (d_points)  cudaFree(d_points);
    if (d_scalars) cudaFree(d_scalars);
    if (d_jac)     cudaFree(d_jac);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "GPU MSM total time: " << us << " us\n";
    return (err == cudaSuccess) ? 0 : 1;
}
