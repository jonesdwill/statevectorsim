import time
import matplotlib.pyplot as plt
from statevectorsim import QuantumState, QuantumCircuit, QuantumGate
from typing import List, Tuple, Dict
import numpy as np

from qiskit.circuit.library import QFT as qiskit_qft
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library import ZGate

# --- Benchmark Circuits ---

def generate_qft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Generates custom QFT circuit (QuantumCircuit.qft).
    """
    return QuantumCircuit.qft(n_qubits, swap_endian=True, inverse=False)

def generate_qpe_circuit(n_estimation: int) -> QuantumCircuit:
    """
    Generates custom QPE circuit using QuantumCircuit.qpe.

    Args:
        n_estimation (int): The size of the counting register (t_qubits).

    Uses U=Z on 1 target qubit (|1> eigenstate, phase 0.5).
    """

    z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    m_qubits = 1

    target_qubit_index = n_estimation
    target_initial_state_gates = [QuantumGate.x(target_qubit_index)]

    return QuantumCircuit.qpe(
        t_qubits=n_estimation,
        unitary_matrix=z_matrix,
        m_qubits=1,
        target_initial_state_gates=target_initial_state_gates
    )


# --- Benchmarking Logic ---

def benchmark_qft(qubit_range: List[int], shots: int, methods_to_run: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Benchmarks the QFT circuit for different qubit counts using the specified methods.
    """
    results: Dict[str, List[Tuple[int, float]]] = {}
    for method in methods_to_run:
        results[method] = []

    for n in qubit_range:

        print(f"\n--- Benchmarking QFT on {n} Qubits (State Space: 2^{n}={2 ** n}) ---")

        # build local circuit simulator
        qc = generate_qft_circuit(n)

        # --- Iterative Benchmarking ---
        for method in methods_to_run:

            # --- Benchmark 'Qiskit Statevector' method using qiskit.quantum_info ---
            if method == 'qiskit_aer':

                total_time_aer = 0.0
                try:
                    qiskit_qft_qc = QiskitCircuit(n)
                    qiskit_qft_qc.append(qiskit_qft(n, do_swaps=True), range(n))

                    for i in range(shots):
                        start_time = time.time()

                        _ = Statevector.from_instruction(qiskit_qft_qc)

                        end_time = time.time()
                        total_time_aer += (end_time - start_time)

                    avg_time_aer = total_time_aer / shots
                    results['qiskit_aer'].append((n, avg_time_aer))

                    print(f"Qiskit Statevector (quantum_info): Avg time over {shots} shots: {avg_time_aer:.6f} s")

                except Exception as e:
                    print(f"WARNING: Qiskit Statevector benchmark failed for N={n}. Error: {e}")

            else:
                # --- Benchmark 'tensor', 'bitmask', 'sparse' methods ---

                total_time = 0.0
                for i in range(shots):

                    state_to_run = QuantumState(n, mode=method)
                    state_to_run.basis_state(0)

                    if method == 'sparse':
                        state_to_run.to_sparse()

                    start_time = time.time()
                    qc.run(state_to_run, method=method)
                    end_time = time.time()
                    total_time += (end_time - start_time)

                avg_time = total_time / shots
                results[method].append((n, avg_time))
                print(f"{method.capitalize()}: Avg time over {shots} shots: {avg_time:.6f} s")

    return results


def benchmark_qpe(qubit_range: List[int], shots: int, methods_to_run: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Benchmarks the QPE circuit for different counting qubit counts using the specified methods.
    """
    results: Dict[str, List[Tuple[int, float]]] = {}
    for method in methods_to_run:
        results[method] = []

    for n in qubit_range:
        n_total = n + 1 # n estimation qubits + 1 target qubit

        print(f"\n--- Benchmarking QPE on {n} Estimation Qubits ({n_total} Total Qubits) ---")

        # build local circuit simulator
        qc = generate_qpe_circuit(n)

        # --- Iterative Benchmarking ---
        for method in methods_to_run:

            if method == 'qiskit_aer':
                # --- Benchmark 'Qiskit Statevector' method using qiskit.quantum_info ---

                total_time_aer = 0.0

                try:
                    qiskit_qpe_qc = QiskitCircuit(n_total)
                    qiskit_qpe_qc.append(PhaseEstimation(n, ZGate()), range(n_total))
                    qiskit_qpe_qc.x(n)

                    for i in range(shots):
                        start_time = time.time()

                        _ = Statevector.from_instruction(qiskit_qpe_qc)

                        end_time = time.time()
                        total_time_aer += (end_time - start_time)

                    avg_time_aer = total_time_aer / shots
                    results['qiskit_aer'].append((n, avg_time_aer))
                    print(f"Qiskit Statevector (quantum_info): Avg time over {shots} shots: {avg_time_aer:.6f} s")

                except Exception as e:
                    print(f"WARNING: Qiskit Statevector benchmark failed for N={n}. Error: {e}")

            else:
                # --- Benchmark 'tensor', 'bitmask', 'sparse' methods ---

                total_time = 0.0
                for i in range(shots):

                    state_to_run = QuantumState(n_total, mode=method)
                    state_to_run.basis_state(0)

                    if method == 'sparse':
                        state_to_run.to_sparse()

                    start_time = time.time()
                    qc.run(state_to_run, method=method)
                    end_time = time.time()
                    total_time += (end_time - start_time)

                avg_time = total_time / shots
                results[method].append((n, avg_time))
                print(f"{method.capitalize()}: Avg time over {shots} shots: {avg_time:.6f} s")

    return results

def plot_benchmarks(results: Dict[str, List[Tuple[int, float]]], shots: int, circuit_title: str):
    """
    Generates a matplotlib plot of simulation time vs. number of qubits.
    """

    all_n = []
    for data in results.values():
        all_n.extend([d[0] for d in data])
    max_qubits = max(all_n) if all_n else 0

    plt.figure(figsize=(10, 6))

    labels = {
        'tensor': 'Tensor',
        'bitmask': 'Bitmask',
        'sparse': 'Sparse Bitmask',
        'qiskit_aer': r"IBM's Qiskit Statevector ($\mathtt{qiskit.quantum\_info}$)"
    }

    for method, data in results.items():
        if not data:
            continue

        n_qubits = [d[0] for d in data]
        times = [d[1] for d in data]
        plt.plot(n_qubits, times, marker='o', linestyle='-', label=labels.get(method, method))

    plt.title(f'{circuit_title} Benchmark Comparison: Up to {max_qubits} Qubits, Avg over {shots} Shots')
    plt.xlabel('Number of Qubits (n)')
    plt.ylabel('Average Simulation Time (seconds) - Log Scale')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if all_n:
        plt.xticks(sorted(list(set(all_n))))

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{circuit_title}_benchmark.png')