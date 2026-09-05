#!/usr/bin/env python3
"""
Add Apple Silicon (MPS) support to the benchmark scripts.

Both scripts select `cuda` or fall back to `cpu`, so on a Mac they run on CPU --
and the MoE script deliberately refuses to, because on CPU it is a multi-day job.
This makes MPS a first-class device.

The edits are deliberately tiny, and the experiment is untouched:

  * The device line gains an MPS branch.
  * The MoE's CPU abort guard accepts MPS.

Nothing else needs changing. The autocast calls are already written as
`torch.autocast("cuda", enabled=(DEV.type == "cuda"))`, so on MPS they disable
themselves and the model trains in fp32 -- which is what you want, since MPS
fp16 support is uneven. GradScaler is already gated the same way, the cuDNN and
TF32 settings are no-ops off CUDA, and `torch.cuda.manual_seed_all` is already
behind an availability check. There are no `.cuda()` calls, no `pin_memory`, and
no `non_blocking` transfers to fix.

One thing this cannot fix, and you should know before comparing anything:
switching device changes floating-point numerics and the RNG stream. Numbers
from an MPS run are NOT bit-comparable with your original CUDA logs. They are a
fresh replication, not a continuation -- which is fine, as long as you do not
mix seeds from the two in one paired test.

Usage:
    python3 enable_mps.py /path/to/regular_degree_cnn_benchmark.py \\
                          /path/to/moe_routing_validation.py
    python3 enable_mps.py --check <files>     # report without writing
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

OLD_DEVICE = 'DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")'

NEW_DEVICE = '''def _select_device():
    """CUDA, else Apple Silicon MPS, else CPU. Set RXD_DEVICE to override."""
    import os as _os
    forced = _os.environ.get("RXD_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


DEV = _select_device()'''

OLD_GUARD = '    if DEV.type == "cuda" or QUICK or os.environ.get("ALLOW_CPU", "0") == "1":'
NEW_GUARD = (
    '    if DEV.type in ("cuda", "mps") or QUICK or os.environ.get("ALLOW_CPU", "0") == "1":'
)

EDITS = [
    ("device selection", OLD_DEVICE, NEW_DEVICE, True),
    ("MoE CPU abort guard", OLD_GUARD, NEW_GUARD, False),
]


def patch(path: pathlib.Path, check_only: bool) -> int:
    source = path.read_text()
    applied, skipped = [], []

    for name, old, new, required in EDITS:
        if new in source:
            skipped.append(f"{name}: already patched")
        elif old in source:
            if source.count(old) != 1:
                print(f"  ERROR {name}: found {source.count(old)} matches, expected exactly 1")
                return 1
            source = source.replace(old, new)
            applied.append(name)
        elif required:
            print(f"  ERROR {name}: anchor not found -- has this file been edited?")
            return 1
        else:
            skipped.append(f"{name}: not present in this file")

    for note in skipped:
        print(f"  skip  {note}")
    for name in applied:
        print(f"  apply {name}")

    if not applied:
        print("  nothing to do")
        return 0

    if check_only:
        print("  (--check: not written)")
        return 0

    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  backup {backup.name}")
    path.write_text(source)

    compile(source, str(path), "exec")  # syntax check before we claim success
    print(f"  wrote  {path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=pathlib.Path)
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    status = 0
    for path in args.files:
        print(f"\n{path}")
        if not path.is_file():
            print("  ERROR file not found")
            status = 1
            continue
        status |= patch(path, args.check)

    if status == 0 and not args.check:
        print(
            "\nDone. Smoke-test each before committing a long run:\n"
            "  QUICK=1 python3 regular_degree_cnn_benchmark.py\n"
            "  QUICK=1 python3 moe_routing_validation.py\n"
            "The QUICK path uses 3 seeds and 3 epochs, so it finishes in minutes and will\n"
            "surface any MPS operator gaps before you commit a night of compute."
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
