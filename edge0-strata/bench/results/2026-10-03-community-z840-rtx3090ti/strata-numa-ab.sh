#!/usr/bin/env bash
# strata-numa-ab.sh — 在 Z840 上对比三种 NUMA 放置方案的 Strata 速度
#
#   A_node0      numactl --cpunodebind=0 --membind=0   （现在 systemd 服务用的方案）
#   B_interleave numactl --interleave=all             （两路 35 核 + 内存交织）
#   C_default    不加 numactl                           （内核默认放置，作基线）
#
# 每种方案：停服务 -> 调整大页分布 -> 启动引擎 -> 预热 -> 测 N 次解码 -> 测 N 次长提示词读取 -> 停引擎
# 结束后把大页恢复到节点 0 并重新启动 systemd 服务。
#
# 用法（在 Z840 上，tmux 里）:   sudo -v && ./strata-numa-ab.sh
# 可选环境变量:  RUNS=5 WARM=2 ORDER="A_node0 B_interleave C_default"

set -u

STRATA_DIR="$HOME/Strata"
RUN="$STRATA_DIR/run-iq3_s.sh"
LOG="$STRATA_DIR/strata-iq3_s.log"
URL="http://127.0.0.1:8080"
KEY="z840-strata-key"
RUNS="${RUNS:-5}"
WARM="${WARM:-2}"
HP0=/sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
HP1=/sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
TOTAL_HP=28000
STAMP="$(date +%Y%m%d-%H%M)"
OUT="$HOME/strata-numa-ab-$STAMP.txt"
ORDER="${ORDER:-A_node0 B_interleave C_default}"

declare -A CMD HP
CMD[A_node0]="numactl --cpunodebind=0 --membind=0"; HP[A_node0]="$TOTAL_HP 0"
CMD[B_interleave]="numactl --interleave=all";         HP[B_interleave]="$((TOTAL_HP/2)) $((TOTAL_HP/2))"
CMD[C_default]="";                                    HP[C_default]="$((TOTAL_HP/2)) $((TOTAL_HP/2))"

DECODE_PROMPT='用 Python 写一个带单元测试的 LRU 缓存类，并逐行解释。'
# 长提示词：一段中英混排文字重复 70 次，约 5K token；每次前面加序号让前缀缓存失效
PARA='Strata 把 24,576 个专家分成三层：最热的常驻显存，全部放内存，查表放 SSD。The expert pool is memory-bandwidth bound, so NUMA placement matters on a dual-socket Haswell. 每个 token 只激活 10 个专家。 '
LONG_BODY="$(python3 -c "import sys; print(sys.argv[1]*70)" "$PARA")"

say(){ printf '%s\n' "$*" | tee -a "$OUT"; }
die(){ say "!! $*"; exit 1; }

need(){ command -v "$1" >/dev/null 2>&1 || die "缺少命令: $1"; }
need numactl; need curl; need python3

hp_set(){  # $1=node0 页数 $2=node1 页数；先减后加，避免瞬时超额
  local n0="$1" n1="$2" c0 c1
  c0=$(cat $HP0); c1=$(cat $HP1)
  if (( n0 < c0 )); then echo "$n0" | sudo tee $HP0 >/dev/null; fi
  if (( n1 < c1 )); then echo "$n1" | sudo tee $HP1 >/dev/null; fi
  echo "$n0" | sudo tee $HP0 >/dev/null
  echo "$n1" | sudo tee $HP1 >/dev/null
  say "   大页分布 node0/node1 = $(cat $HP0)/$(cat $HP1)（要求 $n0/$n1）"
  [[ "$(cat $HP0)" == "$n0" && "$(cat $HP1)" == "$n1" ]] || say "   !! 大页没有分满，内存可能碎片化；结果仍可参考"
}

wait_ready(){  # 最多等 300 s
  local i
  for i in $(seq 1 150); do
    if curl -s -m 3 -H "Authorization: Bearer $KEY" "$URL/health" 2>/dev/null | grep -q '"loaded": true'; then return 0; fi
    sleep 2
  done
  return 1
}

stop_engine(){
  pkill -INT -f 'serve/server.py' 2>/dev/null
  local i
  for i in $(seq 1 60); do
    pgrep -f 'serve/server.py' >/dev/null || break
    sleep 1
  done
  pgrep -f 'serve/server.py' >/dev/null && pkill -9 -f 'serve/server.py'
  pgrep -f 'engine/strata' >/dev/null && pkill -9 -f 'engine/strata'
  sleep 3
}

chat(){  # $1=内容 $2=max_tokens；只关心服务端日志，不保留回答
  python3 - "$1" "$2" <<'PY' > /tmp/strata-ab-req.json
import json, sys
print(json.dumps({"model": "qwen", "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": int(sys.argv[2]), "reasoning_effort": "none", "temperature": 0}))
PY
  curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
       --data-binary @/tmp/strata-ab-req.json "$URL/v1/chat/completions" > /dev/null
}

# 从日志新增行里抽出 (prompt tok/s, decode tok/s, hit%)；参数为测量前的日志行数
parse_new(){
  tail -n +"$(( $1 + 1 ))" "$LOG" | python3 -c '
import re, sys
pp, tg, hit = [], [], []
for line in sys.stdin:
    m = re.search(r"read in [\d.]+ ms \(([\d.]+) tok/s\).*generated in [\d.]+ ms \(([\d.]+) tok/s\)", line)
    if m: pp.append(float(m.group(1))); tg.append(float(m.group(2)))
    m = re.search(r"hit rate: ([\d.]+)%", line)
    if m: hit.append(float(m.group(1)))
import statistics as st
def med(x): return f"{st.median(x):.1f}" if x else "-"
def rng(x): return f"{min(x):.1f}-{max(x):.1f}" if x else "-"
print(f"{med(pp)}|{rng(pp)}|{med(tg)}|{rng(tg)}|{med(hit)}")
'
}

say "=== Strata NUMA A/B  $STAMP ==="
say "机器: $(hostname)  内核: $(uname -r)  引擎: $(grep -m1 -oE 'v0\.[0-9.]+' "$STRATA_DIR/setup.py" 2>/dev/null || echo ?)"
say "每方案: 预热 $WARM 次, 解码测 $RUNS 次(512 tok), 长提示词测 $RUNS 次(约 5K tok)"
say "GPU 挂在 NUMA 节点: $(cat /sys/bus/pci/devices/$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^0000//' | tr 'A-Z' 'a-z')/numa_node 2>/dev/null || echo ?)"
say ""

sudo -n true 2>/dev/null || die "请先运行 sudo -v 让 sudo 缓存密码，再启动本脚本"
say ">> 停止 systemd 服务 strata"
sudo systemctl stop strata
stop_engine

declare -A R_PP R_PPR R_TG R_TGR R_HIT R_START
for name in $ORDER; do
  say ""
  say "########## 方案 $name : ${CMD[$name]:-<不加 numactl>} ##########"
  read -r n0 n1 <<<"${HP[$name]}"
  hp_set "$n0" "$n1"
  cd "$STRATA_DIR" || die "找不到 $STRATA_DIR"
  t0=$(date +%s)
  if [[ -n "${CMD[$name]}" ]]; then
    ${CMD[$name]} "$RUN" > "/tmp/strata-ab-$name.out" 2>&1 &
  else
    "$RUN" > "/tmp/strata-ab-$name.out" 2>&1 &
  fi
  if ! wait_ready; then
    say "   !! 引擎 300 s 内没就绪，跳过。最后输出："; tail -5 "/tmp/strata-ab-$name.out" | tee -a "$OUT"
    stop_engine; continue
  fi
  R_START[$name]=$(( $(date +%s) - t0 ))
  say "   就绪用时 ${R_START[$name]} s；大页空闲: $(grep HugePages_Free /proc/meminfo | awk '{print $2}')"
  say "   引擎内存分布(numastat, MB):"; numastat -p "$(pgrep -f engine/strata | head -1)" 2>/dev/null | tail -3 | sed 's/^/     /' | tee -a "$OUT"
  grep -oE 'pcie_frac [0-9.]+|pool_workers=[0-9]+|hugetlb 2 MB pages|using 4 KB pages' "$LOG" | tail -3 | sed 's/^/   日志: /' | tee -a "$OUT"

  say "   预热 $WARM 次 ..."
  for i in $(seq 1 "$WARM"); do chat "$DECODE_PROMPT" 512; done

  say "   解码测速 $RUNS 次 ..."
  l0=$(wc -l < "$LOG")
  for i in $(seq 1 "$RUNS"); do chat "$DECODE_PROMPT" 512; printf '.'; done; echo
  IFS='|' read -r _ _ tg tgr hit <<<"$(parse_new "$l0")"
  R_TG[$name]="$tg"; R_TGR[$name]="$tgr"; R_HIT[$name]="$hit"
  say "   解码: 中位数 $tg tok/s (范围 $tgr), 命中率 $hit%"

  say "   长提示词测速 $RUNS 次 ..."
  l0=$(wc -l < "$LOG")
  for i in $(seq 1 "$RUNS"); do chat "第 $i 轮，请总结下文要点：$LONG_BODY" 32; printf '.'; done; echo
  IFS='|' read -r pp ppr _ _ _ <<<"$(parse_new "$l0")"
  R_PP[$name]="$pp"; R_PPR[$name]="$ppr"
  say "   提示词读取: 中位数 $pp tok/s (范围 $ppr)"

  stop_engine
done

say ""
say "=================== 汇总 ==================="
printf '%-14s %-10s %-14s %-18s %-10s %-16s %-18s\n' "方案" "就绪(s)" "解码中位 tok/s" "解码范围" "命中率%" "提示词中位 tok/s" "提示词范围" | tee -a "$OUT"
for name in $ORDER; do
  printf '%-14s %-10s %-14s %-18s %-10s %-16s %-18s\n' "$name" "${R_START[$name]:--}" "${R_TG[$name]:--}" "${R_TGR[$name]:--}" "${R_HIT[$name]:--}" "${R_PP[$name]:--}" "${R_PPR[$name]:--}" | tee -a "$OUT"
done
say ""
say ">> 恢复大页到节点 0 并重启 systemd 服务（方案 A 的配置）"
hp_set "$TOTAL_HP" 0
sudo systemctl start strata
say "完成。结果已存到 $OUT"
say "如果方案 B 明显更快(>3%)，改 /etc/systemd/system/strata.service："
say "  ExecStartPre 的大页行改为 node0=14000 node1=14000，ExecStart 改为 numactl --interleave=all ..."
say "  然后 sudo systemctl daemon-reload && sudo systemctl restart strata，并重跑 ./setup.sh --calibrate（在同样的 numactl 下）"
