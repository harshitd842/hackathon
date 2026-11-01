"""
profile_model.py - measure what the model actually costs to run.

This answers the question a compact-AI judge will always ask: "you say it is
lightweight - show me the numbers." It measures, on your own machine:

  * model file size
  * load time
  * process RAM before and after loading the model
  * CPU utilisation during inference
  * latency per image: mean, median, p95, over N runs
  * throughput (images per second)

and writes the results into model_metrics.json, next to the existing entries.
It never overwrites your accuracy figures.

No extra dependencies
---------------------
Only onnxruntime and numpy are required - the same packages the app already
uses. psutil is used if it happens to be installed, but it is not needed:
on Windows the script reads process memory and CPU time through ctypes, and on
Linux/macOS through the standard library. So this runs on a locked-down machine
where pip installs are blocked.

Usage
-----
    python profile_model.py
    python profile_model.py --runs 50
    python profile_model.py --threads 1
"""
import argparse
import ctypes
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# process measurement: psutil if present, otherwise pure stdlib / ctypes
# --------------------------------------------------------------------------
FORCE_NO_PSUTIL = os.environ.get("PROFILE_NO_PSUTIL") == "1"

try:
    if FORCE_NO_PSUTIL:
        raise ImportError
    import psutil  # noqa: F401
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

_IS_WIN = platform.system() == "Windows"


if _HAVE_PSUTIL:

    def _proc():
        return psutil.Process()

    def ram_mb():
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def cpu_time_s():
        t = psutil.Process().cpu_times()
        return t.user + t.system

    def cpu_count():
        return psutil.cpu_count(logical=True)

elif _IS_WIN:
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    # explicit argtypes matter on 64-bit: without them ctypes passes the process
    # handle as a 32-bit int and the Win32 calls fail, silently returning zero
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
    ]
    _k32.GetProcessTimes.restype = wintypes.BOOL

    try:
        _k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        _k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        _get_mem = _k32.K32GetProcessMemoryInfo
    except AttributeError:
        _psapi = ctypes.WinDLL("psapi", use_last_error=True)
        _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        _get_mem = _psapi.GetProcessMemoryInfo

    def ram_mb():
        c = _PMC()
        c.cb = ctypes.sizeof(c)
        if not _get_mem(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError(f"GetProcessMemoryInfo failed (error {ctypes.get_last_error()})")
        return c.WorkingSetSize / (1024 * 1024)

    def _ft(v):
        return ((v.dwHighDateTime << 32) | v.dwLowDateTime) / 1e7

    def cpu_time_s():
        c, e, k, u = (wintypes.FILETIME() for _ in range(4))
        if not _k32.GetProcessTimes(
            _k32.GetCurrentProcess(),
            ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u),
        ):
            raise OSError(f"GetProcessTimes failed (error {ctypes.get_last_error()})")
        return _ft(k) + _ft(u)

    def cpu_count():
        return os.cpu_count() or 1

else:
    import resource

    def ram_mb():
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes
        return usage / 1024.0 if platform.system() != "Darwin" else usage / (1024 * 1024)

    def cpu_time_s():
        u = resource.getrusage(resource.RUSAGE_SELF)
        return u.ru_utime + u.ru_stime

    def cpu_count():
        return os.cpu_count() or 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="crop_disease.onnx")
    ap.add_argument("--runs", type=int, default=50, help="inference runs to time")
    ap.add_argument("--warmup", type=int, default=5, help="untimed runs first")
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    ap.add_argument("--out", default="model_metrics.json")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"error: {args.model} not found. Run this from the project folder.")

    import onnxruntime as ort

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = args.threads

    # ---------- baseline ----------
    base_rss = ram_mb()
    size_mb = os.path.getsize(args.model) / (1024 * 1024)

    # ---------- load ----------
    t0 = time.perf_counter()
    sess = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000
    after_load_rss = ram_mb()

    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    x = np.random.rand(1, 3, args.img, args.img).astype("float32")

    # ---------- warmup ----------
    for _ in range(args.warmup):
        sess.run(None, {inp.name: x})

    # ---------- timed ----------
    times = []
    cpu0 = cpu_time_s()
    wall0 = time.perf_counter()
    for _ in range(args.runs):
        t = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - wall0
    cpu_used = cpu_time_s() - cpu0

    peak_rss = ram_mb()
    cores = cpu_count()
    cpu_pct_total = (cpu_used / wall) * 100.0 if wall > 0 else 0.0
    cpu_pct_one_core = cpu_pct_total / max(1, cores)

    times_sorted = sorted(times)
    p95 = times_sorted[min(len(times_sorted) - 1, int(round(0.95 * (len(times_sorted) - 1))))]

    result = {
        "model_file": args.model,
        "model_size_mb": round(size_mb, 2),
        "input_shape": list(inp.shape),
        "output_shape": list(out.shape),
        "load_time_ms": round(load_ms, 1),
        "ram_before_load_mb": round(base_rss, 1),
        "ram_after_load_mb": round(after_load_rss, 1),
        "ram_delta_mb": round(after_load_rss - base_rss, 1),
        "ram_peak_mb": round(peak_rss, 1),
        "cpu_percent_of_all_cores": round(cpu_pct_total, 1),
        "cpu_percent_of_one_core": round(cpu_pct_one_core, 1),
        "latency_ms_mean": round(statistics.mean(times), 2),
        "latency_ms_median": round(statistics.median(times), 2),
        "latency_ms_p95": round(p95, 2),
        "latency_ms_min": round(min(times), 2),
        "latency_ms_max": round(max(times), 2),
        "throughput_img_per_s": round(args.runs / wall, 1),
        "runs": args.runs,
        "cpu_threads": args.threads or "onnxruntime default",
        "cpu_count": cores,
        "runtime": "onnxruntime " + ort.__version__,
        "platform": platform.platform(),
        "measurement_source": "psutil" if _HAVE_PSUTIL else "stdlib/ctypes (no psutil)",
    }

    print("\n--- measured on this machine ---")
    width = max(len(k) for k in result)
    for k, v in result.items():
        print(f"  {k:<{width}}  {v}")

    # ---------- merge into model_metrics.json ----------
    data = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["profiling"] = result
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nwritten into {args.out} under the key 'profiling'")
    print("accuracy figures were left untouched")

    print("\nPaste this into your README:")
    print(
        f"  Runs in {result['ram_delta_mb']:.0f} MB RAM at "
        f"{result['cpu_percent_of_one_core']:.0f}% of one core, "
        f"{result['latency_ms_median']:.1f} ms per image "
        f"({result['throughput_img_per_s']:.0f} images/sec) "
        f"on {cores} logical core(s). No GPU."
    )


if __name__ == "__main__":
    main()
