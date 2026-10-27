import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def scripts():
    return sorted((ROOT / "reference").glob("*.py"))


def run(path):
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, cwd=ROOT)
    elapsed = time.perf_counter() - start
    lines = [line for line in proc.stdout.splitlines() if "PASS" in line]
    ok = proc.returncode == 0 and bool(lines)
    detail = lines[-1].split("PASS", 1)[1].strip() if ok else (proc.stderr.strip().splitlines() or ["no PASS line"])[-1]
    return ok, elapsed, detail


if __name__ == "__main__":
    paths = scripts()
    if not paths:
        sys.exit("no reference checks found in reference/")
    width = max(len(str(path.relative_to(ROOT))) for path in paths)
    failed = 0
    for path in paths:
        ok, elapsed, detail = run(path)
        failed += not ok
        name = str(path.relative_to(ROOT))
        detail = detail if len(detail) <= 62 else detail[:59] + "..."
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {elapsed:5.2f}s  {detail}")
    print(f"\n{len(paths) - failed}/{len(paths)} checks passed")
    sys.exit(1 if failed else 0)
