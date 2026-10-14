#!/usr/bin/env bash
# Two copies of the wave-2 training snapshot that differ from it in one line only: caft_vllm.runner_mask's row choice.
# all    = every position (prompt included);  prompt = prompt positions before the last prompt token only.
set -euo pipefail
B=/scratch/yelarys.yertaiuly/bestcase-step4; O=$B/source_v2_3968e8cf6e89
L='    mask=positions >= np.repeat(prompt-1,counts)'
[ "$(grep -cxF "$L" $O/verl/verl/utils/caft_vllm.py)" = 1 ] || { echo "mask line not found once"; exit 2; }
for s in all prompt; do
  D=$B/source_leak_$s; [ -d $D ] && { echo "$D exists"; continue; }
  cp -a $O $D; chmod -R u+w $D
  case $s in all) NEW='    mask=positions >= 0';; prompt) NEW='    mask=positions < np.repeat(prompt-1,counts)';; esac
  python3 - "$D/verl/verl/utils/caft_vllm.py" "$L" "$NEW" <<'PY'
import sys; p,old,new=sys.argv[1:]; t=open(p).read(); assert t.count(old+'\n')==1; open(p,'w').write(t.replace(old+'\n',new+'\n'))
PY
  find $D -name __pycache__ -prune -exec rm -rf {} +
  chmod -R a-w $D
  echo "== $s"; diff -r $O $D | head -8
done
