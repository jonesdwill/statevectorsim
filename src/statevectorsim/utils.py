import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from typing import List, Dict, Union

def single_qubit_bloch_vector(state, qubit_index):
    """
    Compute qubit's Bloch vector (x, y, z) in a multi-qubit state.
    """
    n_qubits = int(np.log2(len(state)))
    state_tensor = state.reshape([2] * n_qubits)

    axes_to_trace = tuple(i for i in range(n_qubits) if i != qubit_index)

    rho = np.tensordot(state_tensor, np.conj(state_tensor),
                       axes=(axes_to_trace, axes_to_trace))

    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])

    vec = np.array([x, y, z])

    return vec


def plot_bloch_spheres(state, fig_size=(4, 4), max_cols=4):
    """
    Bloch sphere plot for multi-qubit state, given a statevector. Plots multiple bloch spheres.
    """

    n_qubits = int(np.log2(len(state)))

    n_cols = min(n_qubits, max_cols)
    n_rows = math.ceil(n_qubits / max_cols)

    fig = plt.figure(figsize=(fig_size[0] * n_cols, fig_size[1] * n_rows))

    # Sphere surface mesh
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    # equators for plotting
    eq_u = np.linspace(0, 2*np.pi, 200)
    eq_zero = np.zeros_like(eq_u)

    # XY equator (Z=0)
    eq_x = np.cos(eq_u)
    eq_y = np.sin(eq_u)
    eq_z = eq_zero

    # XZ equator (Y=0)
    eq2_x = np.cos(eq_u)
    eq2_y = eq_zero
    eq2_z = np.sin(eq_u)

    # YZ equator (X=0)
    eq3_x = eq_zero
    eq3_y = np.cos(eq_u)
    eq3_z = np.sin(eq_u)

    # loop over qubits
    for q in range(n_qubits):

        # get bloch vector
        vec = single_qubit_bloch_vector(state, q)

        # create subplot
        ax = fig.add_subplot(n_rows, n_cols, q+1, projection='3d')

        # --- plotting ---
        # sphere surface
        ax.plot_surface(
            xs, ys, zs,
            rstride=1, cstride=1,
            color="white", alpha=0.05,
            edgecolor="gray", linewidth=0.3
        )

        # plot equators
        ax.plot(eq_x, eq_y, eq_z, color='black', linewidth=0.6)
        ax.plot(eq2_x, eq2_y, eq2_z, color='black', linewidth=0.6)
        ax.plot(eq3_x, eq3_y, eq3_z, color='black', linewidth=0.6)

        # plot axis
        ax.plot([-1, 1], [0, 0], [0, 0], color="black", linewidth=0.8)
        ax.plot([0, 0], [-1, 1], [0, 0], color="black", linewidth=0.8)
        ax.plot([0, 0], [0, 0], [-1, 1], color="black", linewidth=0.8)

        # plot bloch-vector
        ax.quiver(
            0, 0, 0,
            vec[0], vec[1], vec[2],
            color='blue',
            linewidth=2,
            arrow_length_ratio=0.2,
            linestyle='-',
            alpha=0.9
        )

        ## --- compute amplitudes for labelling ---
        n_qubits = int(np.log2(len(state)))
        state_tensor = state.reshape([2] * n_qubits)

        # Reduced density matrix for qubit q
        axes_to_trace = tuple(i for i in range(n_qubits) if i != q)
        rho = np.tensordot(state_tensor, np.conj(state_tensor),
                           axes=(axes_to_trace, axes_to_trace))

        # amplitudes
        alpha = np.sqrt(np.real(rho[0, 0]))
        beta = np.sqrt(np.real(rho[1, 1]))
        phase = np.angle(rho[0, 1])
        beta = beta * np.exp(1j * phase)

        ## --- label ---
        title_str = f"Qubit {q}\n$|\\psi\\rangle = {alpha:.2f}|0\\rangle + {abs(beta):.2f}|1\\rangle$"
        ax.set_title(title_str, fontsize=10)

        ax.text(0, 0, 1.5, r"$|0\rangle$", ha='center', va='center', fontsize=10)
        ax.text(0, 0, -1.5, r"$|1\rangle$", ha='center', va='center', fontsize=10)

        label_offset = 1.15
        ax.text(label_offset, 0, 0, 'X', ha='center', va='center', fontsize=10)
        ax.text(0, label_offset, 0, 'Y', ha='center', va='center', fontsize=10)
        ax.text(0, 0, label_offset, 'Z', ha='center', va='center', fontsize=10)

        # --- format ---
        ax.set_box_aspect([1, 1, 1])

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        ax.set_axis_off()
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.view_init(elev=25, azim=40)

    plt.tight_layout()
    plt.show()


def statevector_to_dataframe(state: np.ndarray, little_endian=True):
    """ Convert a statevector to a pandas DataFrame. Little-endian (default). """

    # get num qubits from state
    n = int(np.log2(len(state)))

    # index qubits
    indices = np.arange(len(state))

    # create binary strings for basis states
    if little_endian:
        states = [f"|{i:0{n}b}>" for i in indices]
    else:
        # big-endian: reverse bits
        states = [f"|{format(i, f'0{n}b')[::-1]}>" for i in indices]

    # format amplitudes
    amplitudes = [f"{amp.real:.4f}{'+' if amp.imag >= 0 else '-'}{abs(amp.imag):.4f}j" for amp in state]

    # create dataframe
    df = pd.DataFrame({
        "Index": indices,
        "State": states,
        "Amplitude": amplitudes
    })

    return df


def plot_histogram(results: Dict[str, int], shots: int):
    """ Generates probability histogram from multi-shot measurement results. """

    if not results:
        print("No results to plot.")
        return

    # --- results ---
    # sort keys for consistent plotting order (e.g., '00', '01', '10', '11')
    outcomes = sorted(results.keys())

    # get counts and compute probabilities
    counts = [results.get(o, 0) for o in outcomes]
    probabilities = [c / shots for c in counts]

    # create plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(outcomes, probabilities, alpha=0.7)

    # --- label ---
    plt.title(f'Histogram of Monte-Carlo Simulation ({shots} Shots)', fontsize=14)
    plt.xlabel('Measurement Outcome', fontsize=12)
    plt.ylabel('Probability', fontsize=12)

    # probability text labels
    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_val + 0.01,
                 f'{y_val:.3f}', ha='center', va='bottom')

    # --- format ---
    plt.ylim(0, max(probabilities) * 1.1 or 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)


    plt.show()
