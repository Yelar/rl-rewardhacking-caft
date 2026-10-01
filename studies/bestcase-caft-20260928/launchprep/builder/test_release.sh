#!/usr/bin/env bash
# Tests for release.sh (Linux: reads /proc). Uses only sleep processes and a stub nvidia-smi; no GPU is touched.
# Usage: bash test_release.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d /tmp/reltest-XXXXXXXX); trap 'rm -rf "$T"' EXIT
fail=0; pass=0
ok() { echo "ok - $1"; pass=$((pass + 1)); }
bad() { echo "FAIL - $1"; fail=$((fail + 1)); }
cat > "$T/nvsmi_none" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$T/nvsmi_none"
. "$HERE/release.sh"

# 1. Nothing left: verified at once.
NVSMI="$T/nvsmi_none"
out=$(verify_release "$T/run1" "0" 10); rc=$?
[ $rc = 0 ] && [[ "$out" == "RELEASE VERIFIED after 0 s"* ]] && ok "no run process: verified immediately" || bad "empty case: rc=$rc $out"

# 2. A lingering run process is TERMed at half the deadline; an unrelated process of ours is untouched.
env TMPDIR="$T/run2" setsid sleep 600 &
env TMPDIR="$T/other" setsid sleep 600 &
sleep 1
marked=$(pgrep -u "$(id -u)" -f "sleep 600" | while read -r p; do tr '\0' '\n' < /proc/$p/environ | grep -qxF "TMPDIR=$T/run2" && echo $p; done)
other=$(pgrep -u "$(id -u)" -f "sleep 600" | while read -r p; do tr '\0' '\n' < /proc/$p/environ | grep -qxF "TMPDIR=$T/other" && echo $p; done)
[ -n "$marked" ] && [ -n "$other" ] || bad "setup: marked=$marked other=$other"
[ "$(run_pids "$T/run2")" = "$marked" ] && ok "run_pids finds exactly the run's process" || bad "run_pids: $(run_pids "$T/run2")"
out=$(verify_release "$T/run2" "0" 12); rc=$?
[ $rc = 0 ] && echo "$out" | grep -q "TERM to $marked" && ! kill -0 "$marked" 2>/dev/null && ok "lingering run process TERMed, release verified" || bad "TERM case: rc=$rc $out"
kill -0 "$other" 2>/dev/null && ok "process without the run's TMPDIR untouched" || bad "unrelated process was signalled"

# 3. A run process that ignores TERM is KILLed 30 s later.
env TMPDIR="$T/run3" setsid bash -c 'trap "" TERM; while :; do sleep 1; done' &
sleep 1
out=$(verify_release "$T/run3" "0" 70); rc=$?
[ $rc = 0 ] && echo "$out" | grep -q "RELEASE: KILL" && [ -z "$(run_pids "$T/run3")" ] && ok "TERM-ignoring run process KILLed, release verified" || bad "KILL case: rc=$rc $out"

# 4. One of our processes still on a selected GPU (not the run's): not verified, and it is never signalled.
cat > "$T/nvsmi_busy" <<EOF
#!/usr/bin/env bash
case "\$*" in
  *query-gpu=index,uuid*) echo "7, GPU-fake-7" ;;
  *query-compute-apps*) echo "$other, GPU-fake-7" ;;
esac
EOF
chmod +x "$T/nvsmi_busy"
NVSMI="$T/nvsmi_busy"
[ "$(our_gpu_pids "3,7")" = "7:$other" ] && ok "our_gpu_pids finds our process on a selected GPU (multi-GPU list)" || bad "our_gpu_pids: $(our_gpu_pids "3,7")"
out=$(verify_release "$T/run4" "3,7" 6); rc=$?
[ $rc = 1 ] && echo "$out" | grep -q "RELEASE NOT VERIFIED" && kill -0 "$other" 2>/dev/null && ok "GPU still held: not verified; nothing signalled" || bad "busy GPU case: rc=$rc $out"
[ -z "$(our_gpu_pids "3")" ] && ok "unselected GPU ignored" || bad "unselected GPU counted"

kill "$other" 2>/dev/null
echo "passed $pass, failed $fail"
[ $fail = 0 ]
