#!/bin/bash
# Our own A/A decode script (ab/ab.py, not the community harness), run right after run-harness.sh on the same running
# engine. Two identical arms ('{}' and '{}') are the in-engine A/A control: the engine sees the same request settings
# for both, so any difference between the arms is the method's noise. Round r visits the arms starting at arm r mod 2.
#   c2   = two concurrent 300-token decodes per arm and round, the number is their summed tok/s
#   solo = one 300-token decode per arm and round
# Every request has its own prompt (a salt and the round are in it). ab.py reads the API key from a file, never prints it.
cd /root/prbench
date -u +"ab start %F %T UTC" > ab.log
python3 -u /opt/strata-bmtp-tools/ab.py c2 12 '{}' '{}' > ab-c2.txt 2>&1
echo "c2 exit $?" >> ab.log
python3 -u /opt/strata-bmtp-tools/ab.py solo 12 '{}' '{}' > ab-solo.txt 2>&1
echo "solo exit $?" >> ab.log
date -u +"ab end %F %T UTC" >> ab.log
