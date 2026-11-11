@echo off
rem Config quotidiana (06/10/2026): UD-IQ4_XS, KV int8, YaRN x3 a 786432, cache esperti auto, riserva 1017,
rem --spec 5, --suffix-draft 0, env MT_MIN=1 GATHER=1 PREFETCH=4096 (output riproducibile, +13.8% a 352K da CLI).
rem STRATA_STAGER_THREADS=8 (06/10): prefill +5.9% a 352K rispetto ai 32 thread scelti dal motore.
rem Stessa config sulla 0.1.40.1: run-chat-0.1.40.1.bat. STRATA_PREFILL_CPU_SHARE=0 (09/10): stessi bit della 0.1.40.x sul prefill corto. Encoder immagini: quello della 0.1.39 (compatibile).
title Strata chat quotidiano (UD-IQ4_XS, YaRN x3 768K, KV int8, r1017 p033, spec 5 sp08, nosfx mt1, temp 1, vision CPU, v0.1.41)
cd /d "C:\strata-lab\pr\strata-0.1.41"
"C:\Users\Enky\anaconda3\python.exe" "C:\strata-lab\pr\strata-0.1.41\serve\server.py" "--engine" "strata" "--config" "C:\strata-lab\configs\strata-ud-iq4_xs-chat-yarn3-768k-int8-r1017-p033-sp08-spec5-nosfx-mt1-t1-visioncpu-0.1.41.json" "--port" "8080"
pause
