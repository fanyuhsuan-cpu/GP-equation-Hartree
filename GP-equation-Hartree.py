from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings(
    "ignore", message=r"CuPy failed to preload library.*", category=UserWarning
)
import cudaq
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Edit parameters here, then run:  python cudaq_GP_equ_carleman.py
# The compiled kernel is fixed to N=128, M=4, periodic boundary conditions.
# ---------------------------------------------------------------------------

N = 128
M = 4
L = 20.0
G = 4.0
TOTAL_TIME = 0.5
DT = 0.001
K0 = 2.0
CUDA_Q_TARGET = "nvidia"
CUDA_Q_PRECISION = "fp64"  # Change to "fp32" only if FP64 still exceeds memory.

DX = L / N
STEPS = round(TOTAL_TIME / DT)


# ---------------------------------------------------------------------------
# One self-contained kernel is used for CUDA-Q 0.9.1 compatibility.
# Each of the four particles has a seven-qubit position register.  The
# contact phase is applied directly to the XOR-equality pattern, so no
# persistent or temporary ancilla qubit is required.
# ---------------------------------------------------------------------------

@cudaq.kernel
def hartree_n128_m4_strang(
    input_state: cudaq.State,
    steps: int,
    kinetic_single_z_angles: list[float],
    kinetic_pair_zz_angles: list[float],
    qft_angles: list[float],
    contact_angle: float,
):
    q = cudaq.qvector(input_state)

    for _ in range(steps):
        # kinetic(dt/2), contact(dt), kinetic(dt/2)
        for outer_half in range(2):
            for particle in range(4):
                start = 7 * particle

                # Seven-qubit QFT.
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
                    swap(
                        q[start + swap_index],
                        q[start + 6 - swap_index],
                    )

                # Structured exp(-i k^2 dt/2).  The signed FFT index squared
                # contains only Z and pairwise ZZ terms, so this costs
                # O(log(N)^2) gates rather than an O(N) basis-state lookup.
                for bit in range(7):
                    rz(kinetic_single_z_angles[bit], q[start + bit])

                pair_index = 0
                for first in range(7):
                    for second in range(first + 1, 7):
                        cx(q[start + first], q[start + second])
                        rz(
                            kinetic_pair_zz_angles[pair_index],
                            q[start + second],
                        )
                        cx(q[start + first], q[start + second])
                        pair_index += 1

                # Inverse QFT.
                for swap_index in range(3):
                    swap(
                        q[start + swap_index],
                        q[start + 6 - swap_index],
                    )
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
                for left_particle in range(4):
                    for right_particle in range(left_particle + 1, 4):
                        left = 7 * left_particle
                        right = 7 * right_particle

                        # XOR the two position registers and convert the
                        # equality pattern |0000000> to |1111111>.
                        for bit in range(7):
                            cx(q[left + bit], q[right + bit])
                        for bit in range(7):
                            x(q[right + bit])

                        # Direct seven-body phase: the first six qubits are
                        # controls and the seventh is the R1 target.  This
                        # replaces the former equality ancilla and halves the
                        # FP64 statevector memory at N=128, M=4.
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

                        # Restore both position registers.
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
    """Angles for exp[-i (DT/2) (2*pi/L)^2 m_signed^2]."""

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


def direct_gp_reference(
    orbital0: np.ndarray, h1: np.ndarray
) -> np.ndarray:
    def rhs(_time: float, orbital: np.ndarray) -> np.ndarray:
        return -1j * (h1 @ orbital) + 1j * (G / DX) * np.abs(orbital)**2 * orbital

    solution = solve_ivp(
        rhs,
        (0.0, TOTAL_TIME),
        orbital0,
        method="DOP853",
        t_eval=[TOTAL_TIME],
        rtol=1e-11,
        atol=1e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    final = solution.y[:, -1]
    return final / np.linalg.norm(final)


def product_state(orbital: np.ndarray) -> np.ndarray:
    state = np.asarray([1.0 + 0.0j])
    for _ in range(M):
        state = np.kron(orbital, state)
    return np.ascontiguousarray(state / np.linalg.norm(state))


def one_body_density_matrix(state: np.ndarray) -> np.ndarray:
    tensor = state.reshape((N,) * M, order="F")
    rho = np.zeros((N, N), dtype=np.complex128)
    for particle in range(M):
        matrix = np.moveaxis(tensor, particle, 0).reshape(N, -1)
        rho += matrix @ matrix.conjugate().T / M
    # Retained project convention: rho[p,q] = <a_p^dagger a_q>/M.
    return 0.5 * (rho.T + np.conjugate(rho))


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.linalg.svd(left - right, compute_uv=False)))


def analytical_density(grid: np.ndarray) -> np.ndarray:
    density = 0.5 / np.cosh(grid - 4.0 * TOTAL_TIME)**2
    density /= np.sum(density) * DX
    return density


def save_figures(
    grid: np.ndarray,
    orbital0: np.ndarray,
    density_cudaq: np.ndarray,
    density_gp: np.ndarray,
    density_exact: np.ndarray,
) -> None:
    time_tag = f"{TOTAL_TIME:g}".replace(".", "p")
    initial = np.abs(orbital0)**2 / DX
    cudaq_density = density_cudaq / DX
    gp_density = density_gp / DX

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(grid, initial, ":", lw=1.8, label="Initial")
    axes[0].plot(grid, cudaq_density, "--", lw=2.0, label="quantum method")
    axes[0].plot(grid, gp_density, lw=2.0, label="classical method")
    axes[0].plot(grid, density_exact, ":", lw=2.2, label="Analytical soliton")
    axes[0].set(xlabel="x", ylabel=r"$|\psi(x,t)|^2$",
                title=f"Real-time GP soliton (N={N}, t={TOTAL_TIME:g})")
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
    figure.savefig("gp_density_comparison.png", dpi=180)
    figure.savefig(f"gp_density_comparison_N{N}_t{time_tag}.png", dpi=180)
    plt.close(figure)

    floor = np.finfo(float).tiny
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.semilogy(grid, np.maximum(np.abs(cudaq_density - gp_density), floor),
                   lw=2.0, label="|quantum method - GP|")
    axis.semilogy(grid, np.maximum(np.abs(cudaq_density - density_exact), floor),
                   "--", lw=1.8, label="|quantum method - analytical|")
    axis.semilogy(grid, np.maximum(np.abs(gp_density - density_exact), floor),
                   ":", lw=1.8, label="|classical method - analytical|")
    axis.set(xlabel="x", ylabel="absolute density error",
             title="Pointwise density error")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig("gp_density_error.png", dpi=180)
    figure.savefig(f"gp_density_error_N{N}_t{time_tag}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    if N != 128 or M != 4:
        raise ValueError("the compiled kernel is fixed to N=128, M=4")
    if not np.isclose(STEPS * DT, TOTAL_TIME):
        raise ValueError("TOTAL_TIME must be an integer multiple of DT")

    backend = select_target(CUDA_Q_TARGET)
    grid, orbital0 = grid_and_initial_state()
    h1 = spectral_hamiltonian()

    kinetic_single_z, kinetic_pair_zz = structured_kinetic_angles()
    qft_angles = [float(np.pi / 2**distance) for distance in range(1, 7)]
    contact_angle = G * DT / ((M - 1) * DX)

    started = time.perf_counter()
    output = np.asarray(
        cudaq.get_state(
            hartree_n128_m4_strang,
            cudaq.State.from_data(product_state(orbital0)),
            STEPS,
            kinetic_single_z,
            kinetic_pair_zz,
            qft_angles,
            float(contact_angle),
        )
    ).copy()
    runtime = time.perf_counter() - started

    data_dimension = N**M
    data_state = output[:data_dimension]
    ancilla_leakage = float(np.linalg.norm(output[data_dimension:]))
    rho_cudaq = one_body_density_matrix(data_state)

    orbital_gp = direct_gp_reference(orbital0, h1)
    rho_vector = np.conjugate(orbital_gp)
    rho_gp = np.outer(rho_vector, np.conjugate(rho_vector))

    density_cudaq = np.real(np.diag(rho_cudaq))
    density_gp = np.real(np.diag(rho_gp))
    density_exact = analytical_density(grid)
    save_figures(grid, orbital0, density_cudaq, density_gp, density_exact)

    print()
    print("=" * 60)
    print(f"Finite-M Hartree soliton: N={N}, M={M}, t={TOTAL_TIME:g}")
    print("=" * 60)
    print(f"backend                         = {backend}")
    print(f"runtime_seconds                 = {runtime:.6f}")
    print(f"full_state_norm                 = {np.linalg.norm(output):.12e}")
    print(f"data_state_norm                 = {np.linalg.norm(data_state):.12e}")
    print(f"ancilla_leakage                 = {ancilla_leakage:.12e}")
    print(f"cudaq_vs_GP_trace_distance      = {trace_distance(rho_cudaq, rho_gp):.12e}")
    print(f"density_L1_error_vs_GP          = {np.sum(np.abs(density_cudaq-density_gp)):.12e}")
    print(f"density_L2_error_vs_GP          = {np.linalg.norm(density_cudaq-density_gp):.12e}")
    print(f"density_Linf_error_vs_GP        = {np.max(np.abs(density_cudaq-density_gp)):.12e}")
    print("density_L1_error_vs_analytical  = "
          f"{np.sum(np.abs(density_cudaq/DX-density_exact))*DX:.12e}")
    print("classical_method_L1_vs_analytical      = "
          f"{np.sum(np.abs(density_gp/DX-density_exact))*DX:.12e}")
    time_tag = f"{TOTAL_TIME:g}".replace(".", "p")
    print("figure                          = "
          f"gp_density_comparison_N{N}_t{time_tag}.png")


if __name__ == "__main__":
    main()
