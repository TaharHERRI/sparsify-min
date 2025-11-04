import time, torch

def device_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def measure_latency_ms(fn, *args, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        _ = fn(*args)
    device_synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(*args)
    device_synchronize()
    dt = time.perf_counter() - t0
    return (dt / iters) * 1000.0
