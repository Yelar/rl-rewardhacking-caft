import sys
root = sys.argv[1]
p = root + "/src/train/verl/caft_initialization.py"; s = open(p).read()
old = ("    # The original FSDP actor constructor normalizes 16*16/8 to32 (and 16*16/1 to 256 on one rank).\n"
       "    allowed = ((32, 256) if initialized else (16,)) if v2 else ((32,) if initialized else (16,))\n")
new = ("    # The original FSDP actor constructor normalizes 16*16/8 to32 (and 16*16/1 to 256 on one rank).\n"
       "    # Best-case wave 2: v2 also runs on 2 or 4 ranks per run (16*16/2 = 128, 16*16/4 = 64); virtual-8 balancing\n"
       "    # keeps the 8-rank micro-batch grouping on every one of these layouts.\n"
       "    allowed = ((32, 64, 128, 256) if initialized else (16,)) if v2 else ((32,) if initialized else (16,))\n")
assert s.count(old) == 1, "anchor"
open(p, "w").write(s.replace(old, new))
open(root + "/src/train/verl/test_bestcase_v2_layouts.py", "w").write(open(sys.argv[2]).read())
print("patched")
