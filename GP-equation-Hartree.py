from __future__ import annotations

import gc
import time
import warnings

import numpy as np
warnings.filterwarnings("ignore", message=r"CuPy failed to preload library.*", category=UserWarning)

import cudaq
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------
# M-scaling test derived from the supplied N=128 finite-M Hartree program.
# Run M=2,3,4, keep DT=0.001 and TOTAL_TIME=0.5 (500 iterations), and
# evaluate errors at steps 50,100,...,500.
# -----------------------------------------------------------------------------
N = 128
M_VALUES = (2, 3, 4)
L = 20.0
G = 4.0
TOTAL_TIME = 1.0
DT = 0.001
K0 = 2.0
CHECK_EVERY = 20
CUDA_Q_TARGET = "nvidia"
CUDA_Q_PRECISION = "fp64"

DX = L / N
STEPS = round(TOTAL_TIME / DT)
CHECKPOINTS = tuple(range(CHECK_EVERY, STEPS + 1, CHECK_EVERY))


@cudaq.kernel
def hartree_n128_strang(
    input_state: cudaq.State,
    steps: int,
    m_particles: int,
    kinetic_single_z_angles: list[float],
    kinetic_pair_zz_angles: list[float],
    qft_angles: list[float],
    contact_angle: float,
):
    q = cudaq.qvector(input_state)

    for _ in range(steps):
        # kinetic(dt/2), contact(dt), kinetic(dt/2)
        for outer_half in range(2):
            for particle in range(m_particles):
                start = 7 * particle

                # QFT
                for target_offset in range(7):
                    target = 6 - target_offset
                    h(q[start + target])
                    for delta in range(target):
                        control = target - 1 - delta
                        r1.ctrl(
                            qft_angles[delta],
                            q[start + control],
                            q[start + target],
                        )

                for swap_index in range(3):
                    swap(q[start + swap_index], q[start + 6 - swap_index])

                # structured exp[-i k^2 DT/2]
                for bit in range(7):
                    rz(kinetic_single_z_angles[bit], q[start + bit])

                pair_index = 0
                for first in range(7):
                    for second in range(first + 1, 7):
                        cx(q[start + first], q[start + second])
                        rz(kinetic_pair_zz_angles[pair_index], q[start + second])
                        cx(q[start + first], q[start + second])
                        pair_index += 1

                # inverse QFT
                for swap_index in range(3):
                    swap(q[start + swap_index], q[start + 6 - swap_index])

                for target in range(7):
                    for control in range(target):
                        delta = target - control - 1
                        r1.ctrl(
                            -qft_angles[delta],
                            q[start + control],
                            q[start + target],
                        )
                    h(q[start + target])

            if outer_half == 0:
                for left_particle in range(m_particles):
                    for right_particle in range(left_particle + 1, m_particles):
                        left = 7 * left_particle
                        right = 7 * right_particle

                        # Equality compute: right <- NOT(left XOR right)
                        for bit in range(7):
                            cx(q[left + bit], q[right + bit])
                        for bit in range(7):
                            x(q[right + bit])

                        # Phase on equality subspace.
                        r1.ctrl(
                            contact_angle,
                            q[right],
                            q[right + 1],
                            q[right + 2],
                            q[right + 3],
                            q[right + 4],
                            q[right + 5],
                            q[right + 6],
                        )

                        # Uncompute.
                        for bit in range(7):
                            x(q[right + bit])
                        for bit in range(7):
                            cx(q[left + bit], q[right + bit])


def select_target(requested: str) -> str:
    if CUDA_Q_PRECISION not in ("fp64", "fp32"):
        raise ValueError("CUDA_Q_PRECISION must be 'fp64' or 'fp32'")

    if requested == "nvidia" and CUDA_Q_PRECISION == "fp64":
        candidates = [
            ("nvidia", {"option": "fp64"}, "nvidia:fp64"),
            ("qpp-cpu", {}, "qpp-cpu"),
        ]
    elif requested == "nvidia":
        candidates = [
            ("nvidia", {}, "nvidia:fp32"),
            ("qpp-cpu", {}, "qpp-cpu"),
        ]
    else:
        candidates = [("qpp-cpu", {}, "qpp-cpu")]

    errors = []
    for target, options, label in candidates:
        try:
            cudaq.set_target(target, **options)
            return label
        except RuntimeError as error:
            errors.append(f"{label}: {error}")
    raise RuntimeError("no usable CUDA-Q target; " + "; ".join(errors))


def grid_and_initial_state() -> tuple[np.ndarray, np.ndarray]:
    grid = (np.arange(N) - N / 2) * DX
    orbital = (1.0 / np.cosh(grid)) * np.exp(1j * K0 * grid)
    orbital = orbital.astype(np.complex128)
    orbital /= np.linalg.norm(orbital)
    return grid, orbital


def spectral_hamiltonian() -> np.ndarray:
    momenta = 2.0 * np.pi * np.fft.fftfreq(N, d=DX)
    fourier = np.fft.fft(np.eye(N), axis=0) / np.sqrt(N)
    matrix = fourier.conjugate().T @ np.diag(momenta**2) @ fourier
    return 0.5 * (matrix + matrix.conjugate().T)


def structured_kinetic_angles() -> tuple[list[float], list[float]]:
    c0 = -0.5
    cz = np.asarray(
        [-0.5 * 2**bit for bit in range(6)] + [32.0],
        dtype=np.float64,
    )
    gamma = 0.5 * DT * (2.0 * np.pi / L) ** 2
    single_z = (4.0 * gamma * c0 * cz).tolist()
    pair_zz = [
        float(4.0 * gamma * cz[first] * cz[second])
        for first in range(7)
        for second in range(first + 1, 7)
    ]
    return single_z, pair_zz


def direct_gp_trajectory(orbital0: np.ndarray, h1: np.ndarray) -> dict[int, np.ndarray]:
    times = DT * np.asarray(CHECKPOINTS, dtype=float)

    def rhs(_time: float, orbital: np.ndarray) -> np.ndarray:
        return -1j * (h1 @ orbital) + 1j * (G / DX) * np.abs(orbital)**2 * orbital

    solution = solve_ivp(
        rhs,
        (0.0, TOTAL_TIME),
        orbital0,
        method="DOP853",
        t_eval=times,
        rtol=1e-11,
        atol=1e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    refs = {}
    for idx, step in enumerate(CHECKPOINTS):
        orbital = solution.y[:, idx].copy()
        orbital /= np.linalg.norm(orbital)
        refs[step] = orbital
    return refs


def product_state(orbital: np.ndarray, m_particles: int) -> np.ndarray:
    state = np.asarray([1.0 + 0.0j])
    for _ in range(m_particles):
        state = np.kron(orbital, state)
    return np.ascontiguousarray(state / np.linalg.norm(state))


def one_body_density_matrix(state: np.ndarray, m_particles: int) -> np.ndarray:
    # Conventional RDM rho[p,q]=<p|rho|q>.
    tensor = state.reshape((N,) * m_particles, order="F")
    rho = np.zeros((N, N), dtype=np.complex128)
    for particle in range(m_particles):
        matrix = np.moveaxis(tensor, particle, 0).reshape(N, -1)
        rho += matrix @ matrix.conjugate().T / m_particles
    rho = 0.5 * (rho + rho.conjugate().T)
    rho /= np.trace(rho).real
    return rho


def pure_projector(orbital: np.ndarray) -> np.ndarray:
    orbital = orbital / np.linalg.norm(orbital)
    return np.outer(orbital, np.conjugate(orbital))


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    diff = left - right
    diff = 0.5 * (diff + diff.conjugate().T)
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def dominant_natural_orbital(rho: np.ndarray) -> tuple[float, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = int(np.argmax(eigvals))
    orbital = eigvecs[:, idx]
    orbital /= np.linalg.norm(orbital)
    return float(np.real(eigvals[idx])), orbital


def phase_aligned_l2(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    overlap = float(np.clip(abs(np.vdot(left, right)), 0.0, 1.0))
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * overlap)))


def von_neumann_entropy(rho: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(np.real(eigvals), 0.0, 1.0)
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log(eigvals)))


def analyze_checkpoint(
    state: np.ndarray,
    m_particles: int,
    orbital_gp: np.ndarray,
    keep_density: bool = False,
) -> dict:
    rho = one_body_density_matrix(state, m_particles)
    rho_gp = pure_projector(orbital_gp)

    density_q = np.real(np.diag(rho))
    density_gp = np.abs(orbital_gp)**2

    lambda_max, natural = dominant_natural_orbital(rho)
    purity = float(np.real(np.trace(rho @ rho)))
    depletion = 1.0 - lambda_max
    entropy = von_neumann_entropy(rho)

    overlap = float(np.clip(abs(np.vdot(natural, orbital_gp)), 0.0, 1.0))
    orbital_trace_distance = float(np.sqrt(max(0.0, 1.0 - overlap**2)))

    metrics = {
        "norm": float(np.linalg.norm(state)),
        "purity": purity,
        "lambda_max": lambda_max,
        "depletion": depletion,
        "entropy": entropy,
        "trace_distance": trace_distance(rho, rho_gp),
        "orbital_trace_distance": orbital_trace_distance,
        "orbital_l2": phase_aligned_l2(natural, orbital_gp),
        "density_l1": float(np.sum(np.abs(density_q - density_gp))),
        "density_l2": float(np.linalg.norm(density_q - density_gp)),
        "density_linf": float(np.max(np.abs(density_q - density_gp))),
    }
    if keep_density:
        metrics["density_q"] = density_q
    return metrics


def run_one_m(
    m_particles: int,
    orbital0: np.ndarray,
    gp_refs: dict[int, np.ndarray],
    kinetic_single_z: list[float],
    kinetic_pair_zz: list[float],
    qft_angles: list[float],
    final_densities: dict[int, np.ndarray],
) -> list[list[float]]:
    contact_angle = G * DT / ((m_particles - 1) * DX)

    print()
    print("=" * 145)
    print(f"M={m_particles}: 500 iterations; error evaluated every {CHECK_EVERY} iterations")
    print("=" * 145)
    print(f"contact_angle = {contact_angle:.12e}")
    print(
        " step     t       traceDist      orbL2        depletion"
        "      purity       densityL2     densityL1     densityLinf"
    )

    # Evolve in chained 50-step chunks: exactly 500 total circuit iterations.
    current_state = cudaq.State.from_data(product_state(orbital0, m_particles))
    rows = []
    total_runtime = 0.0

    for step in CHECKPOINTS:
        started = time.perf_counter()

        next_state = cudaq.get_state(
            hartree_n128_strang,
            current_state,
            CHECK_EVERY,
            m_particles,
            kinetic_single_z,
            kinetic_pair_zz,
            qft_angles,
            float(contact_angle),
        )

        del current_state
        current_state = next_state
        gc.collect()

        # Read-only NumPy view/copy for diagnostics.
        state_array = np.asarray(current_state)
        metrics = analyze_checkpoint(
            state_array,
            m_particles,
            gp_refs[step],
            keep_density=(step == STEPS),
        )
        if step == STEPS:
            final_densities[m_particles] = metrics.pop("density_q")

        elapsed = time.perf_counter() - started
        total_runtime += elapsed

        print(
            f"{step:5d}  {step*DT:7.3f}  "
            f"{metrics['trace_distance']:12.5e}  "
            f"{metrics['orbital_l2']:12.5e}  "
            f"{metrics['depletion']:12.5e}  "
            f"{metrics['purity']:11.8f}  "
            f"{metrics['density_l2']:12.5e}  "
            f"{metrics['density_l1']:12.5e}  "
            f"{metrics['density_linf']:12.5e}"
        )
        print(
            f"        norm={metrics['norm']:.12e} "
            f"lambdaMax={metrics['lambda_max']:.12e} "
            f"entropy={metrics['entropy']:.5e} "
            f"orbTraceDist={metrics['orbital_trace_distance']:.5e} "
            f"chunk_runtime={elapsed:.2f}s"
        )

        rows.append([
            m_particles,
            step,
            step * DT,
            metrics["norm"],
            metrics["purity"],
            metrics["lambda_max"],
            metrics["depletion"],
            metrics["entropy"],
            metrics["trace_distance"],
            metrics["orbital_trace_distance"],
            metrics["orbital_l2"],
            metrics["density_l1"],
            metrics["density_l2"],
            metrics["density_linf"],
            elapsed,
        ])

        del state_array
        gc.collect()

    print(f"M={m_particles} total_runtime = {total_runtime:.2f}s")
    return rows


def save_results(rows: list[list[float]]) -> None:
    data = np.asarray(rows, dtype=float)
    header = (
        "M,step,time,norm,purity,lambda_max,depletion,entropy,"
        "trace_distance_rho_vs_gp,orbital_trace_distance,"
        "natural_orbital_phase_aligned_L2,density_L1,density_L2,density_Linf,"
        "checkpoint_runtime_seconds"
    )

    np.savetxt(
        "hartree_M2_M3_M4_error_every50.csv",
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    for m_particles in M_VALUES:
        subset = data[data[:, 0] == m_particles]
        np.savetxt(
            f"hartree_M{m_particles}_error_every50.csv",
            subset,
            delimiter=",",
            header=header,
            comments="",
        )


def print_m_scaling_summary(rows: list[list[float]]) -> None:
    data = np.asarray(rows, dtype=float)

    print()
    print("=" * 125)
    print("M-scaling summary")
    print("=" * 125)
    print("The last three numbers are (M-1)*error for M=2,3,4.")

    metric_columns = (
        ("traceDist", 8),
        ("orbitalL2", 10),
        ("depletion", 6),
    )

    for step in CHECKPOINTS:
        print(f"\nstep={step}, t={step*DT:.3f}")
        for name, col in metric_columns:
            vals = []
            scaled = []
            for m_particles in M_VALUES:
                row = data[(data[:, 0] == m_particles) & (data[:, 1] == step)][0]
                value = row[col]
                vals.append(value)
                scaled.append((m_particles - 1) * value)
            print(
                f"  {name:10s}: "
                f"M2={vals[0]:.6e} M3={vals[1]:.6e} M4={vals[2]:.6e}  "
                f"scaled=({scaled[0]:.6e}, {scaled[1]:.6e}, {scaled[2]:.6e})"
            )


def save_final_density_figures(
    grid: np.ndarray,
    orbital0: np.ndarray,
    density_gp: np.ndarray,
    final_densities: dict[int, np.ndarray],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_tag = f"{TOTAL_TIME:g}".replace(".", "p")
    initial = np.abs(orbital0)**2 / DX
    gp_density = density_gp / DX
    density_exact = 0.5 / np.cosh(grid - 4.0 * TOTAL_TIME)**2
    density_exact /= np.sum(density_exact) * DX

    for m_particles in M_VALUES:
        cudaq_density = final_densities[m_particles] / DX
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))

        axes[0].plot(grid, initial, ":", lw=1.8, label="Initial")
        axes[0].plot(grid, cudaq_density, "--", lw=2.0,
                     label="quantum method")
        axes[0].plot(grid, gp_density, lw=2.0, label="classical method")
        axes[0].plot(grid, density_exact, ":", lw=2.2,
                     label="Analytical soliton")
        axes[0].set(xlabel="x", ylabel=r"$|\psi(x,t)|^2$",
                    title=f"GP equation (M={m_particles}, N={N}, t={TOTAL_TIME:g})")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(grid, cudaq_density - initial, "--", lw=2.0,
                     label="quantum method - initial")
        axes[1].plot(grid, gp_density - initial, lw=2.0,
                     label="classical method - initial")
        axes[1].plot(grid, density_exact - initial, ":", lw=2.2,
                     label="Analytical - initial")
        axes[1].axhline(0.0, color="gray", lw=1.0)
        axes[1].set(xlabel="x", ylabel=r"$\rho(x,t)-\rho(x,0)$",
                    title="Density change")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        figure.tight_layout()
        figure.savefig(
            f"gp_density_comparison_N{N}_M{m_particles}_t{time_tag}.png",
            dpi=180,
        )
        plt.close(figure)


def main() -> None:
    if STEPS != 1000:
        raise ValueError("This test is intended for exactly 500 iterations.")
    if STEPS % CHECK_EVERY != 0:
        raise ValueError("CHECK_EVERY must divide STEPS.")

    backend = select_target(CUDA_Q_TARGET)
    grid, orbital0 = grid_and_initial_state()
    h1 = spectral_hamiltonian()

    print("=" * 80)
    print("Finite-M scaling test")
    print("=" * 80)
    print(f"backend      = {backend}")
    print(f"N            = {N}")
    print(f"M values     = {M_VALUES}")
    print(f"DT           = {DT}")
    print(f"TOTAL_TIME   = {TOTAL_TIME}")
    print(f"STEPS        = {STEPS}")
    print(f"CHECK_EVERY  = {CHECK_EVERY}")

    # GP reference is independent of M, so solve it only once.
    gp_refs = direct_gp_trajectory(orbital0, h1)
    kinetic_single_z, kinetic_pair_zz = structured_kinetic_angles()
    qft_angles = [float(np.pi / 2**distance) for distance in range(1, 7)]

    all_rows = []
    final_densities = {}
    for m_particles in M_VALUES:
        all_rows.extend(
            run_one_m(
                m_particles,
                orbital0,
                gp_refs,
                kinetic_single_z,
                kinetic_pair_zz,
                qft_angles,
                final_densities,
            )
        )
        gc.collect()

    save_results(all_rows)
    print_m_scaling_summary(all_rows)
    save_final_density_figures(
        grid,
        orbital0,
        np.abs(gp_refs[STEPS])**2,
        final_densities,
    )

    print("\nSaved:")
    print("  hartree_M2_M3_M4_error_every50.csv")
    for m_particles in M_VALUES:
        print(f"  hartree_M{m_particles}_error_every50.csv")
        time_tag = f"{TOTAL_TIME:g}".replace(".", "p")
        print(f"  gp_density_comparison_N{N}_M{m_particles}_t{time_tag}.png")


if __name__ == "__main__":
    main()
