// src/artifact/dequant.cpp - the `strata-dequant` CLI (dequantize a GGUF tensor and dump F32).
// The dequantizers themselves are header-only in include/strata/artifact/dequant.hpp.
#include "strata/artifact/dequant.hpp"

#ifdef STRATA_DEQUANT_SELFTEST
int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: dequant <file.gguf>\n");
        return 2;
    }
    try {
        strata::GgufFile g(argv[1]);
        int checked = 0, bad = 0;
        for (const auto& t : g.tensors()) {
            if (t.type != 42) continue; // Q2_0 only for now
            if (checked >= 8) break;
            const uint8_t* base = g.tensor_data(t);
            const uint64_t nblocks = t.elements() / 64;
            for (uint64_t b = 0; b < nblocks && b < 64; ++b) {
                const uint8_t* blk = base + b * 18;
                float out[64];
                strata::dequantize_q2_0(blk, out);
                const float d = strata::fp16_to_fp32(strata::read_u16(blk));
                // STRUCTURAL INVARIANT: every value must be exactly one of {-d, 0, d, 2d}.
                for (int j = 0; j < 64; ++j) {
                    const float v = out[j];
                    const bool ok = (v == -d) || (v == 0.0f) || (v == d) || (v == 2.0f * d) || (d == 0.0f);
                    if (!ok) {
                        ++bad;
                        if (bad < 4)
                            std::printf("  VIOLATION %s block %llu elem %d: v=%g d=%g\n", t.name.c_str(),
                                        (unsigned long long)b, j, v, d);
                    }
                }
            }
            if (checked == 0) {
                // A folded checksum over the FIRST 64 blocks, for cross-checking against the Python
                // implementation on identical bytes. A single implementation being self-consistent
                // proves little - two disagreeing is what finds bugs (round 67).
                double sum = 0, sumabs = 0;
                const uint8_t* base = g.tensor_data(t);
                for (uint64_t b = 0; b < nblocks && b < 64; ++b) {
                    float out[64];
                    strata::dequantize_q2_0(base + b * 18, out);
                    for (int j = 0; j < 64; ++j) {
                        sum += out[j];
                        sumabs += std::fabs(out[j]);
                    }
                }
                std::printf("  CHECKSUM %s sum=%.10g sumabs=%.10g\n", t.name.c_str(), sum, sumabs);
            }
            ++checked;
        }
        // Q5_0 cross-check on the same 64-block window gguf-py is given, using the first Q5_0 tensor.
        for (const auto& t : g.tensors()) {
            if (t.type != 6 && t.type != 20 && t.type != 14 && t.type != 12 && t.type != 13 && t.type != 11 &&
                t.type != 23)
                continue;
            const uint8_t* base = g.tensor_data(t);
            const int bsz =
                (t.type == 20)
                    ? 18
                    : (t.type == 14
                           ? 210
                           : (t.type == 12
                                  ? 144
                                  : (t.type == 13 ? 176 : (t.type == 11 ? 110 : (t.type == 23 ? 136 : 22)))));
            const int nel =
                (t.type == 14 || t.type == 12 || t.type == 13 || t.type == 11 || t.type == 23) ? 256 : 32;
            const uint64_t nb = t.elements() / (uint64_t)nel;
            double sum = 0, sumabs = 0;
            for (uint64_t b = 0; b < nb && b < 64; ++b) {
                float out[256];
                if (t.type == 20)
                    strata::dequantize_iq4_nl(base + b * bsz, out);
                else if (t.type == 14)
                    strata::dequantize_q6_K(base + b * bsz, out);
                else if (t.type == 12)
                    strata::dequantize_q4_K(base + b * bsz, out);
                else if (t.type == 13)
                    strata::dequantize_q5_K(base + b * bsz, out);
                else if (t.type == 11)
                    strata::dequantize_q3_K(base + b * bsz, out);
                else if (t.type == 23)
                    strata::dequantize_iq4_xs(base + b * bsz, out);
                else
                    strata::dequantize_q5_0(base + b * bsz, out);
                for (int j = 0; j < nel; ++j) {
                    sum += out[j];
                    sumabs += std::fabs(out[j]);
                }
            }
            std::printf("  CHECKSUM %s %s sum=%.10g sumabs=%.10g\n", t.type_name(), t.name.c_str(), sum,
                        sumabs);
            float f8[256];
            if (t.type == 20)
                strata::dequantize_iq4_nl(base, f8);
            else if (t.type == 14)
                strata::dequantize_q6_K(base, f8);
            else if (t.type == 12)
                strata::dequantize_q4_K(base, f8);
            else if (t.type == 13)
                strata::dequantize_q5_K(base, f8);
            else if (t.type == 11)
                strata::dequantize_q3_K(base, f8);
            else if (t.type == 23)
                strata::dequantize_iq4_xs(base, f8);
            else
                strata::dequantize_q5_0(base, f8);
            std::printf("  REF first8:");
            for (int j = 0; j < 8; ++j) std::printf(" %.6g", f8[j]);
            std::printf("\n");
            if (t.type == 23) break; // emit Q5_0 first, then IQ4_NL
        }
        std::printf("%s\n  Q2_0 tensors spot-checked: %d\n  symbolic invariant violations: %d -> %s\n",
                    argv[1], checked, bad, bad ? "FAIL" : "PASS");
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("ERROR: %s\n", e.what());
        return 1;
    }
}
#endif
