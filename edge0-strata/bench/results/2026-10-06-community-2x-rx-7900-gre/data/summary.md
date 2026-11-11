
## interleave500 (500 greedy tokens per request, one restart per request)

| arm | runs | prompt tok/s median (range) | decode tok/s median (range) | TTFT s median (range) | wall s median (range) | finish |
| --- | ---: | --- | --- | --- | --- | --- |
| A | 10 | 961.0 (841.5..1270.1) | 69.3 (66.4..74.3) | 2.9 (1.9..3.6) | 10.1 (6.9..10.7) | ['length', 'stop'] |
| B | 10 | 959.9 (839.6..1271.9) | 63.8 (59.7..69.7) | 2.9 (1.9..3.7) | 10.3 (6.0..11.5) | ['length', 'stop'] |
- pair 1: A 66.4 -> B 60.5 tok/s (-8.9%), prompt 926 -> 923 tok/s
- pair 2: A 72.6 -> B 59.7 tok/s (-17.8%), prompt 842 -> 854 tok/s
- pair 3: A 67.8 -> B 68.3 tok/s (+0.7%), prompt 972 -> 972 tok/s
- pair 4: A 68.8 -> B 68.1 tok/s (-1.0%), prompt 1270 -> 1267 tok/s
- pair 5: A 73.2 -> B 63.3 tok/s (-13.5%), prompt 1231 -> 1251 tok/s
- pair 6: A 67.4 -> B 63.5 tok/s (-5.8%), prompt 919 -> 913 tok/s
- pair 7: A 66.9 -> B 63.8 tok/s (-4.6%), prompt 852 -> 840 tok/s
- pair 8: A 71.0 -> B 69.7 tok/s (-1.8%), prompt 950 -> 948 tok/s
- pair 9: A 74.3 -> B 64.6 tok/s (-13.1%), prompt 1270 -> 1272 tok/s
- pair 10: A 69.9 -> B 63.9 tok/s (-8.6%), prompt 1205 -> 1154 tok/s
- B vs A decode change: median -7.2% (-17.8%..+0.7%) over 10 pairs
- prompt tokens: 1757..4238 (median 2913)
- reused: 0..0 (median 0)
- fresh: 1757..4238 (median 2913)
- generated: 256..500 (median 500)

## short arms (bench_prefill.py: 128 tokens, warmup + 4 fresh + 4 follow-up per arm)

| arm | side | kind | runs | prompt tok/s | decode tok/s | prompt read ms | decode ms |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| batch2 | A | fresh | 4 | 1542.7 (1301.4..1712.5) | 79.1 (74.1..82.0) | 4195.5 (3055.0..5172.0) | 1295.5 (926.0..1727.0) |
| batch2 | A | followup | 4 | 227.3 (188.4..311.4) | 70.1 (62.3..71.3) | 668.0 (595.0..796.0) | 1827.0 (1796.0..2055.0) |
| batch2 | B | fresh | 4 | 1404.0 (1155.0..1552.9) | 68.3 (66.2..69.4) | 4665.5 (3348.0..5695.0) | 1874.0 (1844.0..1934.0) |
| batch2 | B | followup | 4 | 288.7 (174.1..300.7) | 62.8 (59.6..69.1) | 833.5 (655.0..877.0) | 2044.0 (1852.0..2147.0) |
| batch4 | A | fresh | 4 | 1452.3 (956.0..1798.1) | 71.2 (56.7..72.3) | 4657.5 (3505.0..5183.0) | 1798.0 (1771.0..2257.0) |
| batch4 | A | followup | 4 | 213.9 (141.3..298.2) | 61.0 (57.8..68.0) | 699.0 (569.0..832.0) | 2096.5 (1883.0..2211.0) |
| batch4 | B | fresh | 4 | 1117.7 (854.2..1238.9) | 68.5 (66.0..74.9) | 6027.5 (4196.0..7166.0) | 1869.5 (1709.0..1939.0) |
| batch4 | B | followup | 4 | 147.1 (144.7..256.3) | 60.9 (59.0..67.6) | 780.0 (771.0..967.0) | 2103.5 (1894.0..2168.0) |
| pipeline-off | A | fresh | 4 | 1582.8 (1302.1..1756.8) | 77.1 (70.9..83.9) | 4129.5 (2962.0..5062.0) | 1663.5 (1526.0..1804.0) |
| pipeline-off | A | followup | 4 | 185.8 (182.3..315.1) | 66.2 (60.0..73.2) | 622.0 (603.0..787.0) | 1933.5 (1750.0..2134.0) |
| pipeline-off | B | fresh | 4 | 1568.0 (1174.4..1738.6) | 70.0 (63.1..73.6) | 4332.0 (2997.0..5100.0) | 1829.5 (1738.0..2029.0) |
| pipeline-off | B | followup | 4 | 246.8 (182.7..313.1) | 62.8 (56.9..64.6) | 708.0 (592.0..819.0) | 2029.5 (1841.0..2249.0) |
| resident | A | fresh | 4 | 1558.5 (1235.2..1736.0) | 72.1 (63.9..78.5) | 4247.0 (3007.0..5143.0) | 1778.5 (968.0..2002.0) |
| resident | A | followup | 4 | 186.6 (181.1..301.3) | 63.4 (61.3..70.1) | 619.0 (598.0..823.0) | 2019.0 (1826.0..2086.0) |
| resident | B | fresh | 4 | 1550.0 (1313.5..1718.9) | 69.0 (58.0..72.1) | 4171.0 (3036.0..5155.0) | 1854.0 (1775.0..2208.0) |
| resident | B | followup | 4 | 243.9 (179.6..312.5) | 64.6 (60.5..66.6) | 714.5 (606.0..823.0) | 1981.0 (1921.0..2115.0) |
| stage-trim-off | A | fresh | 4 | 1582.9 (1229.7..1751.3) | 75.7 (67.6..84.7) | 4232.5 (2965.0..5058.0) | 1690.5 (1512.0..1892.0) |
| stage-trim-off | A | followup | 4 | 186.1 (180.4..193.3) | 73.5 (68.7..79.1) | 612.5 (585.0..632.0) | 1743.5 (1617.0..1863.0) |
| stage-trim-off | B | fresh | 4 | 1103.4 (883.2..1213.6) | 67.6 (57.7..77.1) | 6021.5 (4219.0..7303.0) | 1893.5 (1659.0..2218.0) |
| stage-trim-off | B | followup | 4 | 193.4 (132.3..254.3) | 70.9 (65.1..76.7) | 918.5 (813.0..1006.0) | 1805.5 (1669.0..1967.0) |

## soak

- built_by: scripts/soak_summary.py (soak.py's teardown raised TimeoutExpired waiting for sample_mem.py to notice its stop file)
- minutes: 30.0
- interval_s: 60.0
- samples: 31
- sample_span_s: 1800.2
- prompts: 239
- survived: True
- finish_reasons: {"length": 239}
- wall_s_min_median_max: [2.02, 2.55, 3.23]
- engine_rss_mib: {"first": 10810.3, "peak": 11180.1, "unit": "MiB"}
- ram_used_mib: {"first": 22713.6, "peak": 24009.0, "unit": "MiB"}
- vram_used: {"0000:06:00.0": {"first": 15.004, "peak": 15.054, "unit": "GiB"}, "0000:2d:00.0": {"first": 14.314, "peak": 14.44, "unit": "GiB"}}
- prompt_errors: 0
- fault_lines: 0
- first sample: {'unix_s': '1791285726.13', '0000:06:00.0_vram_used': '16109985792', '0000:2d:00.0_vram_used': '15369101312', 'ram_used_mib': '22713.6', 'engine_rss_mib': '10810.3'}
- last sample : {'unix_s': '1791287526.36', '0000:06:00.0_vram_used': '16164212736', '0000:2d:00.0_vram_used': '15444127744', 'ram_used_mib': '23930.5', 'engine_rss_mib': '11180.1'}

## batch concurrency

- batch2: slots=2 elapsed 28.2 s, hung_slots=[], aggregate_decode_tps=56.6
    slot 0: ttft 2.329 s, wall 19.13 s, finish stop, error None
    slot 1: ttft 6.257 s, wall 21.69 s, finish length, error None
    done: 406 tokens in 19.0 s (24.2 tok/s) finish stop, cancelled False
    done: 500 tokens in 22.0 s (32.4 tok/s) finish length, cancelled False
    engine lines kept: 4
- batch4: slots=4 elapsed 50.4 s, hung_slots=[], aggregate_decode_tps=64.6
    slot 0: ttft 3.217 s, wall 37.85 s, finish stop, error None
    slot 1: ttft 8.154 s, wall 42.03 s, finish length, error None
    slot 2: ttft 13.357 s, wall 43.15 s, finish length, error None
    slot 3: ttft 18.675 s, wall 43.85 s, finish length, error None
    done: 453 tokens in 38.0 s (13.1 tok/s) finish stop, cancelled False
    done: 500 tokens in 42.0 s (14.8 tok/s) finish length, cancelled False
    done: 500 tokens in 43.0 s (16.8 tok/s) finish length, cancelled False
    done: 500 tokens in 44.0 s (19.9 tok/s) finish length, cancelled False
    engine lines kept: 5
