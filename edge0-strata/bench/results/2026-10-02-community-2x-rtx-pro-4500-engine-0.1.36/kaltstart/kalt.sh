#!/bin/bash
# Kaltstart-Vergleich Strata 0.1.36: /mnt/models (Gen4) gegen /home (Gen5). Stellt am Ende den Dienst wieder her.
D=~/messungen/2026-10-02-strata-0.1.36/kaltstart; mkdir -p $D
S=~/ai/Strata-0.1.36
cd $S
cp -n strata-swift-iq3_xxs.json strata-swift-iq3_xxs-mnt.json
sed 's#/mnt/models/models/gguf/Swift-IQ3_XXS/#/home/qni/models-gguf/Swift-IQ3_XXS/#g' strata-swift-iq3_xxs-mnt.json > strata-swift-iq3_xxs-gen5.json
for c in mnt gen5; do
  printf '#!/bin/sh\ncd "%s"\nexec "%s/.venv/bin/python" "%s/serve/server.py" "--engine" "strata" "--config" "%s/strata-swift-iq3_xxs-%s.json" "--port" "8080" "--open"\n' $S $S $S $S $c > run-$c.sh
done
grep -c "models-gguf" strata-swift-iq3_xxs-gen5.json
evict() { python3 - <<'PY'
import os,glob
fs=glob.glob('/mnt/models/models/gguf/Swift-IQ3_XXS/*.gguf')+glob.glob('/home/qni/models-gguf/Swift-IQ3_XXS/*.gguf')+glob.glob('/home/qni/ai/Strata-data/packs/swift-iq3_xxs/**/*',recursive=True)+glob.glob('/home/qni/ai/Strata-data/mtp/**/*',recursive=True)+glob.glob('/home/qni/ai/Strata-data/models/*.gguf')+glob.glob('/home/qni/ai/Strata-0.1.36/engine/*')
n=0
for f in fs:
    if os.path.isfile(f):
        fd=os.open(f,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd); n+=1
print("evicted",n)
PY
}
stopall() {
  tmux kill-session -t strata 2>/dev/null
  pkill -f "[s]erve/server.py" ; pkill -f "[e]ngine/strata"; sleep 3; pkill -9 -f "[s]erve/server.py"; pkill -9 -f "[e]ngine/strata"
  for i in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); p=$(ss -ltn | grep -c ":8080 "); [ "$u" -lt 1000 ] && [ "$p" = 0 ] && return 0; sleep 2; done; echo "WARNUNG GPU $u MiB Port $p"
}
startup() { # $1=mnt|gen5  $2=Nr
  stopall; sync; evict; free -g | sed -n 2p
  t0=$(date +%s.%N)
  tmux new-session -d -s strata -n serve "sh $S/run-$1.sh"
  while true; do
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d '{"model":"swift-1.5-iq3_xxs","messages":[{"role":"user","content":"Say hi."}],"max_tokens":1,"temperature":0}' localhost:8080/v1/chat/completions)
    [ "$code" = 200 ] && break
    [ $(echo "$(date +%s.%N)-$t0 > 400" | bc) = 1 ] && { echo "ZEITUEBERSCHREITUNG"; break; }
    sleep 0.5
  done
  t1=$(date +%s.%N); echo "START $1 lauf$2: $(echo "$t1-$t0" | bc) s bis erste Antwort"
  tmux capture-pane -t strata:serve -p -S -200 > $D/pane-$1-$2.txt
  grep -E "experts loaded|filling|ready" $D/pane-$1-$2.txt | head -3
}
trap 'echo "TRAP: stelle Dienst her"; stopall; tmux new-session -d -s strata -n serve "sh $S/run-gen5.sh"' ERR
echo "=== Rohlesetest (direkt, 8 GiB, ohne Cache) ==="
for f in /mnt/models/models/gguf/Swift-IQ3_XXS/*00002-of-00002.gguf /home/qni/models-gguf/Swift-IQ3_XXS/*00002-of-00002.gguf; do
  echo "$f"; dd if="$f" of=/dev/null bs=16M count=512 iflag=direct 2>&1 | tail -n 1
done
for n in 1 2; do startup mnt $n; startup gen5 $n; done
echo "=== FERTIG, Dienst laeuft zuletzt auf gen5 ==="
