// cuda_msm.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>

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

// Barretenberg G1 affine point at infinity (x sentinel, y=0)
__constant__ uint64_t POINT_AT_INFINITY_X[4] = {
    0x9e10460b6c3e7ea4ULL,
    0xcbc0b548b438e546ULL,
    0xdc2822db40c0ac2eULL,
    0x183227397098d014ULL
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
    0x9e10460b6c3e7ea4ULL, 0xcbc0b548b438e546ULL,
    0xdc2822db40c0ac2eULL, 0x183227397098d014ULL
};
static const uint64_t POINT_AT_INFINITY_Y_HOST[4] = {0,0,0,0};

// Fr (scalar field) modulus r, R^2, ninv — used on HOST to decode scalars
static const uint64_t BN254_FR_MOD_HOST[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t BN254_FR_R2_HOST[4] = {
    0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL
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
    uint64_t s = a + b;
    uint64_t c1 = (s < a);
    uint64_t s2 = s + carry;
    uint64_t c2 = (s2 < s);
    out = s2;
    carry = c1 | c2;
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

    // t = a*b
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            uint64_t lo = a.limbs[i] * b.limbs[j];
            uint64_t hi = __umul64hi(a.limbs[i], b.limbs[j]);
            add64_c(t[i+j],   lo, t[i+j],   carry);
            add64_c(t[i+j+1], hi, t[i+j+1], carry);
            int k = i + j + 2;
            while (carry && k < 8) {
                uint64_t s = t[k] + 1;
                carry = (s == 0);
                t[k] = s;
                ++k;
            }
        }
    }

    // Montgomery reduction
    for (int i = 0; i < 4; ++i) {
        uint64_t m = t[i] * BN254_FQ_NINV;
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            uint64_t lo = m * BN254_FQ_MOD[j];
            uint64_t hi = __umul64hi(m, BN254_FQ_MOD[j]);
            add64_c(t[i+j],   lo, t[i+j],   carry);
            add64_c(t[i+j+1], hi, t[i+j+1], carry);
        }
        int k = i + 5;
        while (carry && k < 8) {
            uint64_t s = t[k] + 1;
            carry = (s == 0);
            t[k] = s;
            ++k;
        }
    }

    r.limbs[0] = t[4]; r.limbs[1] = t[5]; r.limbs[2] = t[6]; r.limbs[3] = t[7];
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

    // Windowed method (MSB→LSB)
    JacobianPoint R; jacobian_set_inf(R);

    for (int win = 63; win >= 0; --win) {
        if (!jacobian_is_inf(R)) {
            jacobian_double(R,R);
            jacobian_double(R,R);
            jacobian_double(R,R);
            jacobian_double(R,R);
        }
        int bit = win * W;
        int limb = bit >> 6;
        int off  = bit & 63;

        uint64_t w = 0;
        if (limb < 4) {
            uint64_t cur = s.limbs[limb] >> off;
            if (off <= 60) w = cur & 0xF;
            else {
                int rem = 64 - off;
                int nxt = W - rem;
                w = cur & ((1u << rem) - 1);
                if (limb + 1 < 4) w |= (s.limbs[limb+1] & ((1u << nxt) - 1)) << rem;
            }
        }
        if (w) {
            if (jacobian_is_inf(R)) R = table[w];
            else jacobian_add(R, R, table[w]);
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
    FieldElement base = z;
    FieldElement res; fq_set_one(res);
    uint64_t e[4];
    uint64_t borrow = 2;
    for (int i=0;i<4;++i) {
        if (BN254_FQ_MOD_HOST[i] >= borrow) { e[i] = BN254_FQ_MOD_HOST[i] - borrow; borrow = 0; }
        else { e[i] = BN254_FQ_MOD_HOST[i] - borrow; borrow = 1; }
    }
    for (int limb=0; limb<4; ++limb) {
        for (int bit=0; bit<64; ++bit) {
            if ( (e[limb] >> bit) & 1ULL ) fq_mul(res, res, base);
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
    void* result,                   // out (AffinePoint or array of AffinePoint)
    const void* points_spans,       // SpanInfo*
    const void* scalars_spans,      // SpanInfo*
    size_t num_spans)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    const SpanInfo* Pspans = static_cast<const SpanInfo*>(points_spans);
    const SpanInfo* Sspans = static_cast<const SpanInfo*>(scalars_spans);

    // FIX 2: respect per-span min length
    std::vector<size_t> counts(num_spans);
    size_t total = 0;
    for (size_t i=0;i<num_spans;++i) {
        counts[i] = std::min(Pspans[i].size, Sspans[i].size);
        total += counts[i];
    }

    // Flatten inputs & decode Fr scalars
    std::vector<AffinePoint>  h_points(total);
    std::vector<FieldElement> h_scalars_std(total);
    {
        size_t k=0;
        for (size_t s=0;s<num_spans;++s) {
            const size_t n = counts[s];
            const AffinePoint* pp = static_cast<const AffinePoint*>(Pspans[s].data_ptr);
            const FieldElement* ss = static_cast<const FieldElement*>(Sspans[s].data_ptr);
            for (size_t i=0;i<n;++i,++k) {
                h_points[k] = pp[i];
                FieldElement s_std; fr_from_mont(s_std, ss[i]);
                h_scalars_std[k] = s_std;
            }
        }
    }

    // Declare output host buffer BEFORE any goto targets
    std::vector<JacobianPoint> h_jac;

    // Device buffers
    AffinePoint*   d_points  = nullptr;
    FieldElement*  d_scalars = nullptr;
    JacobianPoint* d_jac     = nullptr;

    cudaError_t err = cudaSuccess;

    if (total == 0) {
        // Degenerate input, just return zeros/infinities
        if (num_spans == 1) {
            AffinePoint out;
            for (int i=0;i<4;++i) { out.x.limbs[i]=POINT_AT_INFINITY_X_HOST[i]; out.y.limbs[i]=0; }
            memcpy(result, &out, sizeof(AffinePoint));
        } else {
            std::vector<AffinePoint> outs(num_spans);
            for (size_t s=0;s<num_spans;++s) {
                for (int i=0;i<4;++i) { outs[s].x.limbs[i]=POINT_AT_INFINITY_X_HOST[i]; outs[s].y.limbs[i]=0; }
            }
            memcpy(result, outs.data(), num_spans*sizeof(AffinePoint));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::cout << "GPU MSM total time: " << us << " us\n";
        return 0;
    }

    err = cudaMalloc(&d_points,  total * sizeof(AffinePoint));        if (err) { std::cerr<<"cudaMalloc points failed\n"; goto CLEANUP; }
    err = cudaMalloc(&d_scalars, total * sizeof(FieldElement));        if (err) { std::cerr<<"cudaMalloc scalars failed\n"; goto CLEANUP; }
    err = cudaMalloc(&d_jac,     total * sizeof(JacobianPoint));       if (err) { std::cerr<<"cudaMalloc jac failed\n"; goto CLEANUP; }

    err = cudaMemcpy(d_points,  h_points.data(),      total*sizeof(AffinePoint),  cudaMemcpyHostToDevice); if (err) { std::cerr<<"cpy points->dev failed\n"; goto CLEANUP; }
    err = cudaMemcpy(d_scalars, h_scalars_std.data(), total*sizeof(FieldElement), cudaMemcpyHostToDevice); if (err) { std::cerr<<"cpy scalars->dev failed\n"; goto CLEANUP; }

    {
        const int TPB = 256;
        const int blocks = (int)((total + TPB - 1) / TPB);
        msm_kernel<<<blocks, TPB>>>(d_jac, d_points, d_scalars, total);
        err = cudaGetLastError();       if (err) { std::cerr<<"kernel launch failed: "<<cudaGetErrorString(err)<<"\n"; goto CLEANUP; }
        err = cudaDeviceSynchronize();  if (err) { std::cerr<<"kernel exec failed: "<<cudaGetErrorString(err)<<"\n"; goto CLEANUP; }
    }

    // Copy back Jacobians
    h_jac.resize(total);
    err = cudaMemcpy(h_jac.data(), d_jac, total*sizeof(JacobianPoint), cudaMemcpyDeviceToHost);
    if (err) { std::cerr<<"cpy jac->host failed\n"; goto CLEANUP; }

    // Accumulate per span on HOST in Jacobian; convert once to affine
    if (num_spans == 1) {
        JacobianPoint acc; j_set_inf(acc);
        for (size_t i=0;i<total;++i) {
            if (j_is_inf(acc)) acc = h_jac[i];
            else j_add(acc, acc, h_jac[i]);
        }
        AffinePoint out;
        j_to_affine(out, acc);
        memcpy(result, &out, sizeof(AffinePoint));
    } else {
        std::vector<AffinePoint> outs(num_spans);
        size_t k=0;
        for (size_t s=0;s<num_spans;++s) {
            const size_t n = counts[s];
            JacobianPoint acc; j_set_inf(acc);
            for (size_t i=0;i<n;++i,++k) {
                if (j_is_inf(acc)) acc = h_jac[k];
                else j_add(acc, acc, h_jac[k]);
            }
            j_to_affine(outs[s], acc);
        }
        memcpy(result, outs.data(), num_spans*sizeof(AffinePoint));
    }

CLEANUP:
    if (d_points)  cudaFree(d_points);
    if (d_scalars) cudaFree(d_scalars);
    if (d_jac)     cudaFree(d_jac);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "GPU MSM total time: " << us << " us\n";
    return (err==cudaSuccess) ? 0 : 1;
}
