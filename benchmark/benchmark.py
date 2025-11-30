import matplotlib.pyplot as plt
import time
import numpy as np
import math
from typing import List, Tuple, Dict, Callable

from src.statevectorsim import QuantumState, QuantumCircuit, QuantumGate
from src.statevectorsim.quantum_backend import QuantumBackend
from src.statevectorsim.utils import create_random_circuit

# Qiskit Imports
try:
    from qiskit import QuantumCircuit as QiskitCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import QFT as qiskit_qft, PhaseEstimation, ZGate, YGate
except ImportError:
    print("WARNING: Qiskit not found. Qiskit benchmarks will fail.")

# ======================================================
#              HELPER FUNCTIONS
# ======================================================

def plot_benchmarks(
        results: Dict[str, List[Tuple[int, float]]],
        shots: int,
        circuit_title: str,
        cutoff_time: float = None
):
    """
    Generates a matplotlib plot of simulation time vs. number of qubits.
    """
    all_n = []
    for data in results.values():
        all_n.extend([d[0] for d in data])

    if not all_n:
        print("No data to plot.")
        return

    max_qubits = max(all_n)
    min_qubits = min(all_n)

    plt.figure(figsize=(12, 7))

    # --- Plot Cutoff Line ---
    if cutoff_time:
        plt.axhline(
            y=cutoff_time,
            color='black',
            linestyle='--',
            linewidth=2.0,
            label=f'Time Limit ({cutoff_time}s)',
            zorder=10,
            alpha=0.9
        )

    # --- Styles ---
    labels = {
        'dense': 'Dense', 'dense_opt_v1': 'Dense (v1 opt))', 'dense_opt_v2': 'Dense (v2 opt)',
        'sparse': 'Sparse', 'sparse_opt_v1': 'Sparse (v1 opt)', 'sparse_opt_v2': 'Sparse (v2 opt)',
        'hybrid': 'Hybrid (Smart Backend)',
        'qiskit_aer': r"IBM's Qiskit Statevector", 'qiskit_opt': r"IBM's Qiskit Statevector (L3 Opt)"
    }
    styles = {
        'dense': 's-', 'dense_opt_v1': 's--', 'dense_opt_v2': 's:',
        'sparse': 'x-', 'sparse_opt_v1': 'x--', 'sparse_opt_v2': 'x:',
        'hybrid': 'd-', 'qiskit_aer': 'd-', 'qiskit_opt': 'd--'
    }
    colours = {
        'dense': 'red', 'dense_opt_v1': 'red', 'dense_opt_v2': 'red',
        'sparse': 'green', 'sparse_opt_v1': 'green', 'sparse_opt_v2': 'green',
        'hybrid': 'blue', 'qiskit_aer': 'purple', 'qiskit_opt': 'purple'
    }

    for method, data in results.items():
        if not data: continue
        n_qubits = [d[0] for d in data]
        times = [d[1] for d in data]

        # Fallback for custom method names
        style = styles.get(method, 'o--')
        label = labels.get(method, method)
        colour = colours.get(method, 'gray')

        plt.plot(n_qubits, times, style, label=label, color=colour, markersize=6)

    plt.title(f'{circuit_title} Benchmark Comparison: Up to {max_qubits} Qubits, Avg over {shots} Shots')
    plt.xlabel('Number of Qubits (n)')
    plt.ylabel('Average Simulation Time (seconds) - Log Scale')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)

    if all_n:
        unique_n = sorted(list(set(all_n)))
        if len(unique_n) > 20:
            plt.xticks(np.arange(min_qubits, max_qubits + 1, 2))
        else:
            plt.xticks(unique_n)

    plt.yscale('log')
    # Auto-adjust ylim to make sure cutoff line is visible
    if cutoff_time:
        current_ymin, current_ymax = plt.ylim()
        plt.ylim(current_ymin, max(current_ymax, cutoff_time * 1.5))

    plt.tight_layout()
    plt.savefig(f'{circuit_title.replace(" ", "_")}_benchmark.png')
    plt.show()

def _calculate_cleaned_average(times: List[float], trim_percent: float = 0.15) -> float:
    """Helper to remove outliers and return mean."""
    if not times: return 0.0
    if len(times) < 5: return sum(times) / len(times)

    sorted_times = sorted(times)
    trim_count = int(len(times) * trim_percent)

    if trim_count > 0:
        clean = sorted_times[trim_count: -trim_count]
    else:
        clean = sorted_times

    if not clean: clean = sorted_times
    return sum(clean) / len(clean)


def _convert_to_qiskit(custom_qc: QuantumCircuit) -> QiskitCircuit:
    """ Translates a custom QuantumCircuit into a Qiskit QuantumCircuit. """
    qiskit_qc = QiskitCircuit(custom_qc.n)

    for gate in custom_qc.gates:
        name = gate.name.lower();
        t = gate.targets;
        c = gate.controls
        try:
            # Single Qubit
            if name == 'h':
                qiskit_qc.h(t[0])
            elif name == 'x':
                qiskit_qc.x(t[0])
            elif name == 'y':
                qiskit_qc.y(t[0])
            elif name == 'z':
                qiskit_qc.z(t[0])
            elif name == 's':
                qiskit_qc.s(t[0])
            elif name == 'sdg':
                qiskit_qc.sdg(t[0])
            elif name == 't':
                qiskit_qc.t(t[0])
            elif name == 'tdg':
                qiskit_qc.tdg(t[0])

            # Parametric
            elif 'rx' in name:
                qiskit_qc.rx(float(name.split('(')[1].split(')')[0]), t[0])
            elif 'ry' in name:
                qiskit_qc.ry(float(name.split('(')[1].split(')')[0]), t[0])
            elif 'rz' in name:
                qiskit_qc.rz(float(name.split('(')[1].split(')')[0]), t[0])

            # Two Qubit
            elif name == 'cx':
                qiskit_qc.cx(c[0], t[0])
            elif name == 'cy':
                qiskit_qc.cy(c[0], t[0])
            elif name == 'cz':
                qiskit_qc.cz(c[0], t[0])
            elif name == 'swap':
                qiskit_qc.swap(t[0], t[1])

            # Controlled Parametric
            elif 'crp' in name or 'cp' in name:
                qiskit_qc.cp(float(name.split('(')[1].split(')')[0]), c[0], t[0])
            elif 'crx' in name:
                qiskit_qc.crx(float(name.split('(')[1].split(')')[0]), c[0], t[0])
            elif 'cry' in name:
                qiskit_qc.cry(float(name.split('(')[1].split(')')[0]), c[0], t[0])
            elif 'crz' in name:
                qiskit_qc.crz(float(name.split('(')[1].split(')')[0]), c[0], t[0])

            # Multi-Control
            elif 'mcx' in name:
                qiskit_qc.mcx(c, t[0])
            elif 'mcy' in name:
                qiskit_qc.append(YGate().control(len(c)), c + t)
            elif 'mcz' in name:
                qiskit_qc.append(ZGate().control(len(c)), c + t)
        except:
            pass
    return qiskit_qc


# ======================================================
#                 ENGINE
# ======================================================

def run_benchmark_engine(
        qubit_range: List[int],
        circuit_factory: Callable[[int], QuantumCircuit],
        methods: List[str],
        shots: int,
        time_limit: float = 0.1,
        cached: bool = False,  # True for QFT/GHZ, False for Random
        title: str = "Benchmark"
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Universal benchmarking function.
    """
    results = {m: [] for m in methods}
    hybrid_backend = QuantumBackend()
    active_methods = set(methods)
    run_qiskit = any('qiskit' in m for m in methods)

    for n in qubit_range:
        print(f"\n--- {title} on {n} Qubits ---")
        if not active_methods:
            print("All methods timed out. Stopping early.")
            break

        # Data storage for outlier cleaning
        raw_times = {m: [] for m in methods}

        # --- PRE-COMPILATION (Cached Only) ---
        static_qc = None
        static_qiskit_raw = None
        static_qiskit_opt = None
        static_hybrid = None
        static_dense_opt = {}

        if cached:
            try:
                static_qc = circuit_factory(n)

                # Qiskit
                if run_qiskit:
                    try:
                        q_temp = _convert_to_qiskit(static_qc)
                        static_qiskit_raw = q_temp
                        static_qiskit_opt = transpile(q_temp, optimization_level=3)
                    except:
                        pass

                # Hybrid
                if 'hybrid' in methods:
                    static_hybrid = hybrid_backend.optimise_circuit(static_qc)
                    hybrid_backend.analyze_mode(static_qc)

                # Dense/Sparse Opt
                for m in methods:
                    if '_opt' in m and 'qiskit' not in m:
                        strategy = 'v1' if 'v1' in m else 'v2'
                        qc_copy = static_qc.copy()
                        qc_copy.optimise(method=strategy)
                        static_dense_opt[m] = qc_copy

            except Exception as e:
                print(f"Error building static circuit for N={n}: {e}")
                continue

        # --- SHOT LOOP ---
        for _ in range(shots):

            # 1. GET CIRCUIT
            if cached:
                current_qc = static_qc
            else:
                current_qc = circuit_factory(n)

            # 2. QISKIT SETUP (Dynamic Only)
            q_raw = static_qiskit_raw
            if not cached and run_qiskit and any('qiskit' in m for m in active_methods):
                try:
                    q_raw = _convert_to_qiskit(current_qc)
                except:
                    pass

            # 3. RUN METHODS
            for method in list(active_methods):
                try:
                    t_start = time.time()

                    # --- Qiskit ---
                    if method == 'qiskit_aer' and q_raw:
                        _ = Statevector.from_instruction(q_raw)

                    elif method == 'qiskit_opt' and (q_raw or static_qiskit_opt):
                        if cached:
                            _ = Statevector.from_instruction(static_qiskit_opt)
                        elif q_raw:
                            # For random, we usually TIME execution only, assuming compile is separate step
                            t_prog = transpile(q_raw, optimization_level=3)
                            t_start = time.time()
                            _ = Statevector.from_instruction(t_prog)

                    # --- Hybrid ---
                    elif method == 'hybrid':
                        qc_to_run = static_hybrid if cached else hybrid_backend.optimise_circuit(current_qc)
                        if not cached: hybrid_backend.analyze_mode(current_qc)

                        st = QuantumState(current_qc.n);
                        st.basis_state(0)

                        t_start = time.time()
                        hybrid_backend.execute(qc_to_run, st, shots=1, inplace=True)

                    # --- Dense / Sparse ---
                    else:
                        mode = 'sparse' if method.startswith('sparse') else 'dense'

                        if cached and method in static_dense_opt:
                            qc_to_run = static_dense_opt[method]
                        elif not cached and ('_opt' in method):
                            strat = 'v1' if 'v1' in method else 'v2'
                            qc_to_run = current_qc.copy()
                            qc_to_run.optimise(method=strat)
                        else:
                            qc_to_run = current_qc

                        st = QuantumState(current_qc.n, mode=mode)
                        if mode == 'sparse': st.to_sparse()
                        st.basis_state(0)

                        t_start = time.time()
                        qc_to_run.run(st, method=mode)

                    # Record Time
                    raw_times[method].append(time.time() - t_start)

                except Exception:
                    pass

        # --- PROCESS RESULTS ---
        for method in list(active_methods):
            avg = _calculate_cleaned_average(raw_times[method])
            if avg > 0:
                results[method].append((n, avg))
                print(f"{method}: {avg:.6f} s")
                if avg > time_limit:
                    print(f"  -> CUTOFF: {method} > limit. Dropping.")
                    active_methods.remove(method)

    return results


# ======================================================
#            FACTORIES TO BENCHMARK
# ======================================================

def qft_factory(n):
    return QuantumCircuit.qft(n, swap_endian=True)

def random_factory(n):
    return create_random_circuit(n, depth=50, bias_factor=0.3)

def ghz_factory(n):
    return QuantumCircuit.ghz(n)

def grover_factory(n):
    marked = 2 ** (n - 1)
    return QuantumCircuit.grover_search(n, marked)

def qpe_factory(n):
    z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
    return QuantumCircuit.qpe(n, z_mat, m_qubits=1, target_initial_state_gates=[QuantumGate.x(n)])

def shors_factory(n: int) -> QuantumCircuit:
    # Map n_qubits -> (Number to factor N, guess a)
    cases = {
        12: (15, 7),  # 4 bits (15) * 3 = 12 qubits
        15: (21, 2),  # 5 bits (21) * 3 = 15 qubits
        18: (33, 5),  # 6 bits (33) * 3 = 18 qubits (35 is also 6 bits)
        21: (65, 2)  # 7 bits (65) * 3 = 21 qubits
    }

    if n not in cases:
        raise ValueError(f"No Shor case defined for n={n}. Valid keys: {list(cases.keys())}")

    N_val, a_val = cases[n]
    return QuantumCircuit.shors(N_val, a_val)

# ======================================================
#               MAIN EXECUTION BLOCK
# ======================================================

if __name__ == "__main__":
    # Global Config
    METHODS = [
        'dense', 'dense_opt_v2',
        'sparse', 'sparse_opt_v2',
        'hybrid',
        'qiskit_aer', 'qiskit_opt'
    ]
    SHOTS = 20
    LIMIT = 0.2  # Cutoff time (seconds)

    # --- 1. Random Circuit ---
    res_rand = run_benchmark_engine(list(range(2, 16)), random_factory, METHODS, SHOTS, LIMIT, cached=False, title="Random Circuit")
    plot_benchmarks(res_rand, SHOTS, "Random Circuit", LIMIT)

    # --- 2. QFT ---
    res_qft = run_benchmark_engine(list(range(2, 16)), qft_factory, METHODS, SHOTS, LIMIT, cached=True, title="QFT")
    plot_benchmarks(res_qft, SHOTS, "QFT", LIMIT)

    # --- 3. Grover ---
    res_grover = run_benchmark_engine(list(range(2, 14)), grover_factory, METHODS, SHOTS, LIMIT, cached=True,title="Grover")
    plot_benchmarks(res_grover, SHOTS, "Grover", LIMIT)

    # --- 4. QPE ---
    res_qpe = run_benchmark_engine(list(range(2, 12)), qpe_factory, METHODS, SHOTS, LIMIT, cached=True, title="QPE")
    plot_benchmarks(res_qpe, SHOTS, "QPE", LIMIT)

    # --- 5. GHZ ---
    res_ghz = run_benchmark_engine(list(range(2, 22)), ghz_factory, METHODS, SHOTS, LIMIT, cached=True, title="GHZ State")
    plot_benchmarks(res_ghz, SHOTS, "GHZ", LIMIT)

    # --- 6. Shors ---
    SHORS_RANGE = [12, 15, 18]

    res_shors = run_benchmark_engine(
        SHORS_RANGE,
        shors_factory,
        METHODS,
        shots=5,
        time_limit=LIMIT,
        cached=True,
        title="Shor's Algorithm"
    )

    plot_benchmarks(res_shors, 5, "Shor's Algorithm", LIMIT)
