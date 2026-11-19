import os
import sys
import json
import math
import hashlib
import subprocess
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15

PROC_DIMS = [2, 2]
NY_LOCAL = GLOBAL_NY // PROC_DIMS[0]  # 256 owned rows per rank
NX_LOCAL = GLOBAL_NX // PROC_DIMS[1]  # 256 owned columns per rank
DX = 1.0 / (GLOBAL_NX + 1)  # 1/513: owned cells are global 1..512, Dirichlet walls sit at 0 and 513
DY = 1.0 / (GLOBAL_NY + 1)
CFL_SAFETY = 0.8  # dt = 0.8 * dx^2 / (4 * kappa_max)
REPORT_EVERY = 100
PACK_THREADS = 128

# Segment order inside the 1D halo buffers. A strip sent towards direction d carries tag d,
# so the receiver expects the tag of the opposite direction.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
OPPOSITE = (DOWN, UP, RIGHT, LEFT)


def _abort_on_error(exc_type, exc, tb):
    sys.__excepthook__(exc_type, exc, tb)
    MPI.COMM_WORLD.Abort(1)  # one crashed rank must not leave the other three blocked in a collective


sys.excepthook = _abort_on_error


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    cart = comm.Create_cart(dims=PROC_DIMS, periods=[False, False], reorder=False)
    cart_rank = cart.Get_rank()
    nbr_up, nbr_down = cart.Shift(0, 1)  # axis 0 = rows (y)
    nbr_left, nbr_right = cart.Shift(1, 1)  # axis 1 = columns (x)
    return SimpleNamespace(
        cart=cart,
        rank=cart_rank,
        coords=tuple(cart.Get_coords(cart_rank)),
        nbrs=(nbr_up, nbr_down, nbr_left, nbr_right),
        gpu_uuid=gpu_uuid,
        seed=instance_seed,
    )


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization (rank-local only)
# ==============================================================================
def initialize_subdomains(ctx):
    row_rank, col_rank = ctx.coords
    # Same seed on every rank -> every rank draws identical pulse parameters -> one consistent global field.
    rng = np.random.default_rng(ctx.seed)
    n_pulses = 3
    pulse_x = rng.uniform(0.2, 0.8, n_pulses)
    pulse_y = rng.uniform(0.2, 0.8, n_pulses)
    pulse_amp = rng.uniform(1.0, 2.0, n_pulses)
    pulse_sigma = rng.uniform(0.04, 0.08, n_pulses)
    phase_x, phase_y = rng.uniform(0.0, 2.0 * np.pi, 2)
    src_x, src_y = rng.uniform(0.3, 0.7, 2)

    # Local index l = 0..257 is global index rank_coord * 256 + l, so x = gj * dx covers [0, 1].
    y = (row_rank * NY_LOCAL + np.arange(NY_LOCAL + 2)) * DY
    x = (col_rank * NX_LOCAL + np.arange(NX_LOCAL + 2)) * DX
    X, Y = np.meshgrid(x, y)  # (258, 258): [i, j] -> (y_i, x_j)

    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * X + phase_x) * np.cos(2.0 * np.pi * Y + phase_y)
    u = np.zeros_like(X)
    for cx, cy, amp, sig in zip(pulse_x, pulse_y, pulse_amp, pulse_sigma):
        u += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig * sig))
    src = 0.05 * np.exp(-((X - src_x) ** 2 + (Y - src_y) ** 2) / (2.0 * 0.03**2))

    nbr_up, nbr_down, nbr_left, nbr_right = ctx.nbrs
    if nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0
    if nbr_down == MPI.PROC_NULL:
        u[-1, :] = 0.0
    if nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if nbr_right == MPI.PROC_NULL:
        u[:, -1] = 0.0
    return u, kappa, src


# ==============================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels & Pinned MPI Exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, send, n):
    k = cuda.grid(1)
    if k < n:
        send[UP * n + k] = u[1, k + 1]  # first owned row
        send[DOWN * n + k] = u[n, k + 1]  # last owned row
        send[LEFT * n + k] = u[k + 1, 1]  # first owned column (stride 258 doubles)
        send[RIGHT * n + k] = u[k + 1, n]  # last owned column


@cuda.jit
def unpack_halos_kernel(u, recv, n):
    k = cuda.grid(1)
    if k < n:
        u[0, k + 1] = recv[UP * n + k]
        u[n + 1, k + 1] = recv[DOWN * n + k]
        u[k + 1, 0] = recv[LEFT * n + k]
        u[k + 1, n + 1] = recv[RIGHT * n + k]


def exchange_halos_async(st, u, stream):
    """Stage A + MPI half of Stage B: GPU pack -> async D2H into pinned memory -> post Irecv/Isend."""
    n = st.n
    pack_halos_kernel[st.pack_blocks, PACK_THREADS, stream](u, st.d_send, n)
    st.d_send.copy_to_host(st.h_send, stream=stream)
    stream.synchronize()  # MPI reads h_send from the host, so the D2H DMA must have landed
    reqs = [
        st.cart.Irecv([st.h_recv[d * n:(d + 1) * n], MPI.DOUBLE], source=st.nbrs[d], tag=OPPOSITE[d])
        for d in range(4)
    ]
    reqs += [
        st.cart.Isend([st.h_send[d * n:(d + 1) * n], MPI.DOUBLE], dest=st.nbrs[d], tag=d)
        for d in range(4)
    ]
    return reqs


def complete_halos(st, u, reqs, stream):
    """Halo half of Stage C: Waitall -> async H2D from pinned memory -> GPU unpack into ghosts 0 and n+1.

    A PROC_NULL receive never touches its zero-filled segment, so wall ghosts are re-written with the
    Dirichlet value u = 0 without any boundary branch.
    """
    MPI.Request.Waitall(reqs)
    st.d_recv.copy_to_device(st.h_recv, stream=stream)
    unpack_halos_kernel[st.pack_blocks, PACK_THREADS, stream](u, st.d_recv, st.n)


# ==============================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil & Dual-Stream Orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, src, r_start, r_end, c_start, c_end, dt, inv_dx2, inv_dy2, lam):
    tile = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    sy = ty + 1
    sx = tx + 1

    # Stage only the box plus its 1-cell apron: the deep-interior launch then never reads ghost
    # cells, which stream_boundary may be unpacking at the same moment.
    row_ok = i <= r_end + 1
    col_ok = j <= c_end + 1
    if row_ok and col_ok:
        tile[sy, sx] = u[i, j]
        if ty == 0:
            tile[0, sx] = u[i - 1, j]
        if tx == 0:
            tile[sy, 0] = u[i, j - 1]
    if ty == TILE_DIM - 1 and i <= r_end and col_ok:
        tile[sy + 1, sx] = u[i + 1, j]
    if tx == TILE_DIM - 1 and j <= c_end and row_ok:
        tile[sy, sx + 1] = u[i, j + 1]
    cuda.syncthreads()

    if i <= r_end and j <= c_end:
        uc = tile[sy, sx]
        kc = kappa[i, j]
        div_y = (0.5 * (kappa[i + 1, j] + kc) * (tile[sy + 1, sx] - uc)
                 - 0.5 * (kappa[i - 1, j] + kc) * (uc - tile[sy - 1, sx])) * inv_dy2
        div_x = (0.5 * (kappa[i, j + 1] + kc) * (tile[sy, sx + 1] - uc)
                 - 0.5 * (kappa[i, j - 1] + kc) * (uc - tile[sy, sx - 1])) * inv_dx2
        u_new[i, j] = uc + dt * (div_x + div_y - lam * uc * uc * uc + src[i, j])


def launch_stencil(st, u, u_new, box, stream):
    r_start, r_end, c_start, c_end = box
    grid = ((c_end - c_start + TILE_DIM) // TILE_DIM, (r_end - r_start + TILE_DIM) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        u, u_new, st.kappa, st.src, r_start, r_end, c_start, c_end, st.dt, st.inv_dx2, st.inv_dy2, LAMBDA_DAMP
    )


def step_serialized(st, u, u_new):
    """Reference: full halo exchange first, then one stencil over all owned cells, single stream."""
    reqs = exchange_halos_async(st, u, st.stream_boundary)
    complete_halos(st, u, reqs, st.stream_boundary)
    launch_stencil(st, u, u_new, st.full_box, st.stream_boundary)
    st.stream_boundary.synchronize()


def step_overlapped(st, u, u_new):
    reqs = exchange_halos_async(st, u, st.stream_boundary)  # Stage A (+ posts Irecv/Isend)
    launch_stencil(st, u, u_new, st.interior_box, st.stream_interior)  # Stage B: needs no ghosts
    complete_halos(st, u, reqs, st.stream_boundary)  # Stage C
    for box in st.ring_boxes:
        launch_stencil(st, u, u_new, box, st.stream_boundary)
    st.stream_interior.synchronize()
    st.stream_boundary.synchronize()


# ==============================================================================
# TASK 5: Distributed Reductions, PTX Audit & Comparative Benchmark
# ==============================================================================
def global_diagnostics(cart, u, u_prev):
    """Global energy sum(u)*dx*dy and L2 residual ||u^n - u^(n-1)|| via Allreduce (no field gather)."""
    a = u.copy_to_host()[1:-1, 1:-1]
    b = u_prev.copy_to_host()[1:-1, 1:-1]
    local = np.array([a.sum() * DX * DY, ((a - b) ** 2).sum() * DX * DY])
    total = np.empty(2)
    cart.Allreduce(local, total, op=MPI.SUM)
    return float(total[0]), math.sqrt(total[1])


def run_mode(st, step_fn, steps):
    u = cuda.to_device(st.u0)
    u_new = cuda.to_device(st.u0)  # wall ghosts start at 0 in both buffers; the stencil never writes ghosts
    trace = [(0, *global_diagnostics(st.cart, u, u_new))]
    start, stop = cuda.event(), cuda.event()
    st.cart.Barrier()
    start.record(st.stream_boundary)
    for n in range(1, steps + 1):
        step_fn(st, u, u_new)
        u, u_new = u_new, u
        if n % REPORT_EVERY == 0:
            trace.append((n, *global_diagnostics(st.cart, u, u_new)))
    stop.record(st.stream_boundary)
    stop.synchronize()
    ms = st.cart.allreduce(cuda.event_elapsed_time(start, stop), op=MPI.MAX)
    return u.copy_to_host(), ms, trace


def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    ptx = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    counts = {op: ptx.count(op) for op in ("st.shared.f64", "ld.shared.f64", "bar.sync")}
    return counts, all(c > 0 for c in counts.values())


def run_solver(github_user: str) -> bool:
    ctx = init_distributed_context(github_user)
    cart, rank = ctx.cart, ctx.rank

    if rank == 0:
        dev = cuda.get_current_device()
        print("=" * 78)
        print(f"GitHub user      : {github_user}")
        print(f"GPU UUID         : {ctx.gpu_uuid}")
        print(f"INSTANCE_SEED    : {ctx.seed} (0x{ctx.seed:016x})")
        print(f"GPU              : CC {dev.compute_capability}, {dev.MULTIPROCESSOR_COUNT} SMs, "
              f"{dev.ASYNC_ENGINE_COUNT} async copy engines")
        print(f"MPI ranks        : {cart.Get_size()} in a {PROC_DIMS[0]}x{PROC_DIMS[1]} Cartesian grid, "
              f"periods=[False, False], MPI.PROC_NULL={MPI.PROC_NULL}")
        print("=" * 78, flush=True)
    for r in range(cart.Get_size()):
        if r == rank:
            up, down, left, right = ctx.nbrs
            print(f"[rank {rank}] coords={ctx.coords} up={up} down={down} left={left} right={right} "
                  f"local=(258, 258) owned=(256, 256)", flush=True)
        cart.Barrier()

    u0, kappa, src = initialize_subdomains(ctx)
    kappa_max = cart.allreduce(float(kappa[1:-1, 1:-1].max()), op=MPI.MAX)
    kappa_min = cart.allreduce(float(kappa[1:-1, 1:-1].min()), op=MPI.MIN)
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max

    n = NX_LOCAL
    st = SimpleNamespace(
        cart=cart,
        nbrs=ctx.nbrs,
        n=n,
        u0=u0,
        dt=dt,
        inv_dx2=1.0 / (DX * DX),
        inv_dy2=1.0 / (DY * DY),
        kappa=cuda.to_device(kappa),
        src=cuda.to_device(src),
        d_send=cuda.device_array(4 * n, dtype=np.float64),
        d_recv=cuda.device_array(4 * n, dtype=np.float64),
        h_send=cuda.pinned_array(4 * n, dtype=np.float64),
        h_recv=cuda.pinned_array(4 * n, dtype=np.float64),
        stream_boundary=cuda.stream(),
        stream_interior=cuda.stream(),
        pack_blocks=(n + PACK_THREADS - 1) // PACK_THREADS,
        full_box=(1, n, 1, n),
        interior_box=(2, n - 1, 2, n - 1),
        ring_boxes=((1, 1, 1, n), (n, n, 1, n), (2, n - 1, 1, 1), (2, n - 1, n, n)),
    )
    st.h_send[:] = 0.0
    st.h_recv[:] = 0.0  # pinned memory starts uninitialised; PROC_NULL segments must read as u = 0

    if rank == 0:
        print(f"dx = dy = 1/{GLOBAL_NX + 1} = {DX:.9e}")
        print(f"kappa range      : [{kappa_min:.6f}, {kappa_max:.6f}]")
        print(f"dt_max (CFL)     : dx^2 / (4 kappa_max) = {dt_max:.6e}")
        print(f"dt used          : {CFL_SAFETY} * dt_max = {dt:.6e}  ({STEPS} steps -> t = {STEPS * dt:.6e})",
              flush=True)

    run_mode(st, step_serialized, 1)  # JIT warm-up so compilation is not timed
    run_mode(st, step_overlapped, 1)

    ser_u, ser_ms, ser_trace = run_mode(st, step_serialized, STEPS)
    ovl_u, ovl_ms, ovl_trace = run_mode(st, step_overlapped, STEPS)
    parity = cart.allreduce(float(np.abs(ser_u - ovl_u).max()), op=MPI.MAX)
    ptx_counts, ptx_ok = ptx_audit()
    parity_ok = parity < 1e-12

    if rank == 0:
        for name, trace in (("serialized", ser_trace), ("overlapped", ovl_trace)):
            print(f"\nAllreduce diagnostics ({name}):")
            print(f"{'step':>6} {'energy E':>22} {'L2 residual':>22}")
            for s, e, r in trace:
                print(f"{s:>6} {e:>22.15e} {r:>22.15e}")
        print(f"\nserialized_ms    : {ser_ms:.3f}  ({ser_ms / STEPS:.4f} ms/step)")
        print(f"overlapped_ms    : {ovl_ms:.3f}  ({ovl_ms / STEPS:.4f} ms/step)")
        print(f"overlap speedup  : {ser_ms / ovl_ms:.3f}x")
        print(f"parity_max_diff  : {parity:.3e}  -> {'PASS' if parity_ok else 'FAIL'} (< 1e-12)")
        print(f"PTX audit        : {ptx_counts} -> {'PASS' if ptx_ok else 'FAIL'}")

        manifest = {
            "github_username": github_user,
            "gpu_uuid": ctx.gpu_uuid,
            "instance_seed": ctx.seed,
            "mpi_ranks": cart.Get_size(),
            "cart_dims": PROC_DIMS,
            "cart_periods": [False, False],
            "global_interior_shape": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": list(u0.shape),
            "tile_dim": TILE_DIM,
            "shared_tile_dim": SHARED_DIM,
            "steps": STEPS,
            "dx": DX,
            "dt": dt,
            "kappa_min": kappa_min,
            "kappa_max": kappa_max,
            "dt_max_cfl": dt_max,
            "lambda_damp": LAMBDA_DAMP,
            "energy_trace": [{"step": s, "energy": e, "l2_residual": r} for s, e, r in ovl_trace],
            "serialized_ms": ser_ms,
            "overlapped_ms": ovl_ms,
            "overlap_speedup": ser_ms / ovl_ms,
            "timing_method": "CUDA events on stream_boundary around the 400-step loop, max over ranks",
            "parity_max_diff": parity,
            "parity_passed": parity_ok,
            "ptx_counts": ptx_counts,
            "ptx_audit_passed": ptx_ok,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("\nWrote submission_manifest.json", flush=True)
    return parity_ok and ptx_ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: mpirun -np 4 python3 distributed_cuda_solver.py <github_username>")
    sys.exit(0 if run_solver(sys.argv[1]) else 1)
