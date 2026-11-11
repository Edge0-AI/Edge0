# Metal kernel parity suite — Apple M4, real Metal device
# repo: 04cedad9d4f0a78492596b1a180626781dbbdad7
# device: Apple M4 apple9 metal4, recommended working set 11.8 GiB, measured read BW up to 96.1 GB/s
# command: cmake -S metal -B build-metal-kern -DSTRATA_METAL_ENGINE=OFF && ninja -C build-metal-kern && ./build-metal-kern/<suite>

gemv_parity            PASS ( parity lines)
100                    PASS ( parity lines)
mma_parity             PASS ( parity lines)
91                     PASS ( parity lines)
dense_mm_parity        PASS ( parity lines)
67                     PASS ( parity lines)
gdn_prompt_parity      PASS ( parity lines)
29                     PASS ( parity lines)
pair_parity            PASS ( parity lines)
26                     PASS ( parity lines)
fuse_parity            PASS ( parity lines)
17                     PASS ( parity lines)
gr2_parity             PASS ( parity lines)
13                     PASS ( parity lines)
gr_parity              PASS ( parity lines)
7                      PASS ( parity lines)
qsa_prompt_parity      PASS ( parity lines)
6                      PASS ( parity lines)
gdn_parity             PASS ( parity lines)
5                      PASS ( parity lines)
qsa_parity             PASS ( parity lines)
5                      PASS ( parity lines)
prompt_parity          PASS ( parity lines)
5                      PASS ( parity lines)
expert_parity          PASS ( parity lines)
1                      PASS ( parity lines)
route_parity           PASS ( parity lines)
1                      PASS ( parity lines)
sample_parity          PASS ( parity lines)
1                      PASS ( parity lines)
moe_mm_parity          FAIL: 'no function mm_gu_42_16_64_16_2' (exit 3) -- Q2_0 expert matmul tile not instantiated
arena_budget_test      PASS (25 ok, includes a 16 GB Mac row)
coop_layout            PASS

SUMMARY: 27 of 28 suites pass on real Metal hardware.
