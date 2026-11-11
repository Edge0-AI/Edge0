"""Sample host RAM in use (total - available), swap and GPU state once per second (Windows, psutil + nvidia-smi)."""
import json, subprocess, time, psutil
while True:
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    try:
        g = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
    except Exception as e:
        g = repr(e)
    print(json.dumps({"epoch_s": time.time(), "host_used_gib": round((vm.total - vm.available) / 2**30, 3),
                      "host_total_gib": round(vm.total / 2**30, 3), "swap_used_gib": round(sw.used / 2**30, 3), "gpu": g}), flush=True)
    time.sleep(1)
