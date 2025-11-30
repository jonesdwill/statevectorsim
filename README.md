# Quantum Simulator

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![PyPI version](https://img.shields.io/badge/version-0.1.0-blue)

A lightweight, quantum state-vector simulator, written in Python. This package includes a backend that
switches between dense (NumPy) and sparse (SciPy CSR) matrix representations, optimised circuit compilation, noise modelling via Monte Carlo methods, and built-in implementations of 
major quantum algorithms. Generally, it can efficiently handle up to 26-qubit circuits, though this is dependent on the memory you have available and circuit sparsity. After this memory scaling becomes too large.

---

## Features
* **Dual Storage** matrix representation of $2^N$ amplitudes:
    - Dense (NumPy) storage.
    - Sparse (SciPy) storage.
  
  This project relies on the C-based efficiency of both packages to do most of the heavy memory lifting.


* **Circuit Compilation:** Options for optimising.
    - V1: Adjacent gate 'fusion' and clutter removal.
    - V2: Uses gate-commutativity rules to swap gates and fuse them, reducing circuit depth. 


* **Smart Backend:** Auto-selects optimisation and storage method by estimating sparsity. Some heuristics gained from testing are used here.


* **Noise Modelling:** Support for Monte Carlo noisy simulations (Depolarizing, Bit-Flip, Phase-Flip channels).


* **Visualisation:** Tools for Bloch sphere representation, ASCII circuit drawing, and measurement histograms. 


* **Circuit&Gate Library:** pre-built implementations of Shor's Algorithm, Grover's search, QPE, QFT, and Quantum Addition (mod N). as well as most common gates.
---

## Project Structure

### statevectorsim package:
* **`QuantumState`**: State manager supporting dense and sparse statevector representations. Handles initialisation, basis-state preparation, and measurement collapse.
* **`QuantumGate`**: Comprehensive library of standard, rotational, controlled, and multi-controlled gates. Gate application logic using Tensor Slicing (dense) and Bitmasking (sparse).
* **`QuantumCircuit`**: Build, optimise, and run circuits. Includes:
    * **Compiler**: Basic gate fusion (V1) and commutative 'lookahead' gate fusion (V2).
    * **Algorithm Library**: Pre-built circuits for Shor's Algorithm, Grover's Search, QPE, QFT, and Quantum Adders.
* **`QuantumNoise`**: Monte-Carlo noise simulation with Depolarizing, Bit-Flip, and Phase-Flip error channels.
* **`QuantumBackend`**: Dispatcher that auto-selects the optimal execution strategy based on circuit topology and qubit count.
* **`utils`**: Visualisation tools including 3D Bloch Sphere plotter, ASCII circuit drawing, Pandas dataframe export, and measurement probability histograms.

### Bonus tools:
* **`test_suite`**: Dedicated file for testing all implemented circuits and gates.
* **`benchmark`**: Dedicated file for benchmarking gate-application and optimiser methods. Includes qiskit circuit transpiler.
---

## Benchmarking

The simulator is benchmarked against IBM's Qiskit Statevector simulator (v2.2). Statevectorsim generally keeps up with or out-performs 
Qiskit. Though this is largely down to the fact that Qiskit breaks down the circuit into something parsable by hardware (Qiskit has a very quick C-foundation). Advantage is amplified when benchmarking for a large number of qubits and dense circuits such as Grover's algorithm. 
Largest source of cost predictably comes from scaling qubits - compiling has some weight-saving on circuit depth, but it wavers in comparison to improving memory usage. 

Below is the benchmark for 100 50-gate random n-qubit circuits.
More benchmarks can be found in ```\benchmark```. 

![Random_Circuit_benchmark.png](benchmark/Random_Circuit_benchmark.png)
---

## Usage

1. Basic Circuit & Measurement
``` python
from statevectorsim.quantum_circuit import QuantumCircuit
from statevectorsim.quantum_gate import QuantumGate
from statevectorsim.quantum_backend import QuantumBackend
from statevectorsim.utils import plot_histogram

# 1. Initialize Backend
backend = QuantumBackend()

# 2. Create a Bell State Circuit
qr = QuantumState(2)
qc = QuantumCircuit(2)
qc.add_gate(QuantumGate.h(0))
qc.add_gate(QuantumGate.cx(0, 1))

# 3. Execute (Smart Backend automagically handles Mode)
# Runs 1024 shots
results = backend.run(qc, qr, shots=1024)

# 4. Visualize
print(f"Counts: {results}")
plot_histogram(results, shots=1024)
```
2. Optimiser 

``` python
qc = QuantumCircuit(3)
# ... add complex sequence of gates ...

print("Original Gate Count:", len(qc.gates))

# Apply V2 Optimization (Commutativity-Aware Fusion)
qc.optimise(method='v2')

print("Optimized Gate Count:", len(qc.gates))
```

3. Pre-built Algorithms
``` python
# Generate Grover's Search for a 4-qubit space, searching for state |13>
grover_circuit = QuantumCircuit.grover_search(n_qubits=4, marked_state_index=13)

backend = QuantumBackend()
final_state = backend.run(grover_circuit, shots=1) # Get statevector

# Print probabilities
print(final_state.get_probabilities())
```

4. Noise
``` python
from statevectorsim.quantum_noise import NoiseModel

# Create a noise model with 1% Depolarizing noise
noise_model = NoiseModel(default_error_rate=0.01)

# Run circuit with noise injection
results = backend.run(qc, shots=1000, noise_model=noise_model)
```

---

## Installation

To set-up, follow the steps:

1. Clone the repository:

``` bash
git clone https://github.com/jonesdwill/Quantum-Simulator.git
cd Quantum-Simulator
```

2. Set up a virtual environment (recommended): If you don't have it already, install **`venv`**:
``` bash
py -m pip install venv
```
Navigate to project directory. Create and activate virtual environment.
``` bash
py -m venv venv
venv\Scripts\activate
```
3. Install Required Packages
``` bash
pip install -r requirements.txt
```
---
## Gate Look-up (`QuantumGate`)

Static methods available in the `QuantumGate` class.

| Category | Gate | Method Signature | Description |
| :--- |:---| :--- | :--- |
| **Standard** | Pauli-X (NOT) | `x(targets)` | Bit flip. |
| | Pauli-Y | `y(targets)` | Phase and bit flip. |
| | Pauli-Z | `z(targets)` | Phase flip on $|1\rangle$. |
| | Hadamard | `h(targets)` | Creates superposition. |
| | Identity | `i(targets)` | No change. |
| **Phase** | Phase (S) | `s(targets)` | $Z$ rotation by $\pi/2$. |
| | Inverse Phase ($S^\dagger$) | `sdg(targets)` | $Z$ rotation by $-\pi/2$. |
| | $\pi/8$ (T) | `t(targets)` | $Z$ rotation by $\pi/4$. |
| | Inverse $\pi/8$ ($T^\dagger$) | `tdg(targets)` | $Z$ rotation by $-\pi/4$. |
| **Rotation** | $R_x(\theta)$ | `rx(targets, theta)` | Rotation about X-axis. |
| | $R_y(\theta)$ | `ry(targets, theta)` | Rotation about Y-axis. |
| | $R_z(\theta)$ | `rz(targets, theta)` | Rotation about Z-axis. |
| **Controlled** | CNOT (CX) | `cx(control, target)` | Controlled-X. |
| | Controlled-Y (CY) | `cy(control, target)` | Controlled-Y. |
| | Controlled-Z (CZ) | `cz(control, target)` | Controlled-Z. |
| | SWAP | `swap(q1, q2)` | Swaps two qubit states. |
| | Controlled- $R_x(\theta)$ (CRX) | `crx(c, t, theta)` | Controlled-Rotation-X. |
| | Controlled- $R_y(\theta)$ (CRY) | `cry(c, t, theta)` | Controlled-Rotation-Y. |
| | Controlled- $R_z(\theta)$ (CRZ) | `crz(c, t, theta)` | Controlled-Rotation-Z. |
| | Controlled-Phase (CRP) | `crp(c, t, theta)` | Controlled-Phase rotation. |
| | Controlled-U | `cu(c, targets, U, k)` | Applies $U^k$ controlled by `c`. |
| **Multi-Controlled**| Toffoli (CCX/MCX) | `mcx(controls, target)`| Multi-Controlled X. |
| | MCY | `mcy(controls, target)`| Multi-Controlled Y. |
| | MCZ | `mcz(controls, target)`| Multi-Controlled Z. |
---

## Circuit Look-up (`QuantumCircuit`)

The `QuantumCircuit` class includes static methods to quickly generate common n-qubit circuits.

| Circuit Name | Method Signature | Qubit Count | Description                                                                                                      |
| :---: | :--- | :---: |:-----------------------------------------------------------------------------------------------------------------|
| **Bell State** | `QuantumCircuit.bell()` | 2 | Creates the Bell state $(\vert 00 \rangle + \vert 11 \rangle)/ \sqrt{2}$.                                        |
| **GHZ State** | `QuantumCircuit.ghz(n_qubits: int)` | $n$ | Creates the $n$-qubit Greenberger–Horne–Zeilinger state $(\vert 0...0 \rangle + \vert 1...1 \rangle)/ \sqrt{2}$. |
| **Grover Search** | `QuantumCircuit.grover_search(n_qubits: int, marked_state_index: int)` | $n$ | Full Grover's algorithm circuit (Oracle + Diffuser) to find the marked index.                                    |
| **QFT** | `QuantumCircuit.qft(n_qubits: int, swap_endian: bool = False)` | $n$ | Performs Quantum Fourier Transform (QFT).                                                                        |
| **IQFT** | `QuantumCircuit.qft(n_qubits: int, inverse=True)` | $n$ | Performs Inverse Quantum Fourier Transform (IQFT).                                                               |
| **QFT Adder** | `QuantumCircuit.qft_adder(n_qubits: int)` | $2n$ | Performs addition: $\vert A \rangle \vert B \rangle \to \vert A \rangle \vert B+A \pmod{2^n} \rangle$.           |
| **QPE** | `QuantumCircuit.qpe(t_qubits: int, unitary_matrix, m_qubits: int)` | $t+m$ | Quantum Phase Estimation to estimate the phase of a unitary operator $U$.                                        |
| **Shor's Algo** | `QuantumCircuit.shors(N: int, a: int)` | $\approx 3 \log N$ | Quantum subroutine for Shor's Algorithm (factoring $N$).                                                         |

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

---

## Version

Current version: 0.1.0
