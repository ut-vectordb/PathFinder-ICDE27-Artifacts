#include <iostream>
#include "distance.h"


namespace ANNS {

    std::unique_ptr<DistanceHandler> get_distance_handler(const std::string& data_type, const std::string& dist_fn) {
        if (data_type == "float") {
            if (dist_fn == "L2")
                return std::make_unique<FloatL2DistanceHandler>();
            else if (dist_fn == "IP") {
                std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
                exit(-1);
            } else if (dist_fn == "cosine") {
                std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
                exit(-1);
            } else {
                std::cerr << "Error: invalid distance function: " << dist_fn << " and data type: " << data_type << std::endl;
                exit(-1);
            }
        } else if (data_type == "int8") {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
            exit(-1);
        } else if (data_type == "uint8") {
            if (dist_fn == "L2")
                return std::make_unique<UInt8L2DistanceHandler>();
            else if (dist_fn == "IP") {
                std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
                exit(-1);
            } else if (dist_fn == "cosine") {
                std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
                exit(-1);
            } else {
                std::cerr << "Error: invalid distance function: " << dist_fn << " and data type: " << data_type << std::endl;
                exit(-1);
            }
        } else {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
            exit(-1);
        }
    }

    // float L2 distance
    float FloatL2DistanceHandler::compute(const char *a, const char *b, IdxType dim) const {
        const float *x = reinterpret_cast<const float *>(a);
        const float *y = reinterpret_cast<const float *>(b);
        
        // naive
        // float ans = 0;
        // for (IdxType i = 0; i < dim; i++) 
        //     ans += (x[i] - y[i]) * (x[i] - y[i]);
        // return ans;

        // simd
        // x = (const float *)__builtin_assume_aligned(x, 32);
        // y = (const float *)__builtin_assume_aligned(y, 32);
        // float ans = 0;
        // #pragma omp simd reduction(+ : ans) aligned(x, y : 32)
        // for (int32_t i = 0; i < (int32_t)dim; i++)
        //     ans += (x[i] - y[i]) * (x[i] - y[i]);
        // return ans;

        //  AVX-2
        __m256 msum0 = _mm256_setzero_ps();

        while (dim >= 8) {
            __m256 mx = _mm256_loadu_ps(x);
            x += 8;
            __m256 my = _mm256_loadu_ps(y);
            y += 8;
            const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
            msum0 = _mm256_add_ps(msum0, _mm256_mul_ps(a_m_b1, a_m_b1));
            dim -= 8;
        }

        __m128 msum1 = _mm256_extractf128_ps(msum0, 1);
        __m128 msum2 = _mm256_extractf128_ps(msum0, 0);
        msum1 = _mm_add_ps(msum1, msum2);

        if (dim >= 4) {
            __m128 mx = _mm_loadu_ps(x);
            x += 4;
            __m128 my = _mm_loadu_ps(y);
            y += 4;
            const __m128 a_m_b1 = _mm_sub_ps(mx, my);
            msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
            dim -= 4;
        }

        if (dim > 0) {
            __m128 mx = masked_read(dim, x);
            __m128 my = masked_read(dim, y);
            __m128 a_m_b1 = _mm_sub_ps(mx, my);
            msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
        }

        msum1 = _mm_hadd_ps(msum1, msum1);
        msum1 = _mm_hadd_ps(msum1, msum1);
        return _mm_cvtss_f32(msum1);

        // AVX-512
        // __m512 msum0 = _mm512_setzero_ps();

        // while (dim >= 16) {
        //     __m512 mx = _mm512_loadu_ps(x);
        //     x += 16;
        //     __m512 my = _mm512_loadu_ps(y);
        //     y += 16;
        //     const __m512 a_m_b1 = mx - my;
        //     msum0 += a_m_b1 * a_m_b1;
        //     dim -= 16;
        // }

        // __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
        // msum1 += _mm512_extractf32x8_ps(msum0, 0);

        // if (dim >= 8) {
        //     __m256 mx = _mm256_loadu_ps(x);
        //     x += 8;
        //     __m256 my = _mm256_loadu_ps(y);
        //     y += 8;
        //     const __m256 a_m_b1 = mx - my;
        //     msum1 += a_m_b1 * a_m_b1;
        //     dim -= 8;
        // }

        // __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        // msum2 += _mm256_extractf128_ps(msum1, 0);

        // if (dim >= 4) {
        //     __m128 mx = _mm_loadu_ps(x);
        //     x += 4;
        //     __m128 my = _mm_loadu_ps(y);
        //     y += 4;
        //     const __m128 a_m_b1 = mx - my;
        //     msum2 += a_m_b1 * a_m_b1;
        //     dim -= 4;
        // }

        // if (dim > 0) {
        //     __m128 mx = masked_read(dim, x);
        //     __m128 my = masked_read(dim, y);
        //     __m128 a_m_b1 = mx - my;
        //     msum2 += a_m_b1 * a_m_b1;
        // }

        // msum2 = _mm_hadd_ps(msum2, msum2);
        // msum2 = _mm_hadd_ps(msum2, msum2);
        // return _mm_cvtss_f32(msum2);
    }

    __m128 FloatL2DistanceHandler::masked_read(IdxType dim, const float *x) {
        __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
        switch (dim) {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
        }
        return _mm_load_ps(buf);
    }

    // Credit: ChatGPT
    float UInt8L2DistanceHandler::compute(const char *a, const char *b, IdxType dim) const {
        const uint8_t *x = reinterpret_cast<const uint8_t *>(a);
        const uint8_t *y = reinterpret_cast<const uint8_t *>(b);

#if 0
        float ans = 0.0f;
        for (IdxType i = 0; i < dim; i++) {
            int diff = static_cast<int>(x[i]) - static_cast<int>(y[i]);
            ans += static_cast<float>(diff * diff);
        }
        return ans;

#else
        __m256i acc = _mm256_setzero_si256();

        // process 32 bytes per iteration
        while (dim >= 32) {
            __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x));
            __m256i vy = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y));

            // split into low / high 16 bytes
            __m128i vx_lo = _mm256_castsi256_si128(vx);
            __m128i vx_hi = _mm256_extracti128_si256(vx, 1);
            __m128i vy_lo = _mm256_castsi256_si128(vy);
            __m128i vy_hi = _mm256_extracti128_si256(vy, 1);

            // widen to int16
            __m256i x_lo_16 = _mm256_cvtepu8_epi16(vx_lo);
            __m256i x_hi_16 = _mm256_cvtepu8_epi16(vx_hi);
            __m256i y_lo_16 = _mm256_cvtepu8_epi16(vy_lo);
            __m256i y_hi_16 = _mm256_cvtepu8_epi16(vy_hi);

            // subtract
            __m256i d_lo = _mm256_sub_epi16(x_lo_16, y_lo_16);
            __m256i d_hi = _mm256_sub_epi16(x_hi_16, y_hi_16);

            // square and widen to int32
            __m256i d_lo_sq = _mm256_madd_epi16(d_lo, d_lo);
            __m256i d_hi_sq = _mm256_madd_epi16(d_hi, d_hi);

            // accumulate
            acc = _mm256_add_epi32(acc, d_lo_sq);
            acc = _mm256_add_epi32(acc, d_hi_sq);

            x += 32;
            y += 32;
            dim -= 32;
        }

        // horizontal reduction of acc
        __m128i acc_lo = _mm256_castsi256_si128(acc);
        __m128i acc_hi = _mm256_extracti128_si256(acc, 1);
        acc_lo = _mm_add_epi32(acc_lo, acc_hi);
        acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
        acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);

        float result = static_cast<float>(_mm_cvtsi128_si32(acc_lo));

        // scalar tail
        for (IdxType i = 0; i < dim; i++) {
            int diff = static_cast<int>(x[i]) - static_cast<int>(y[i]);
            result += static_cast<float>(diff * diff);
        }

        return result;

#endif
    }
}