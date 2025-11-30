import numpy as np
from typing import Dict, Union, List, Optional
from .quantum_circuit import QuantumCircuit
from .quantum_state import QuantumState
from .quantum_noise import NoiseModel


class QuantumBackend:
    """
    Intelligent dispatcher that separates compilation (optimization) from execution.
    """

    def __init__(self):
        # Heuristic thresholds
        self.HARD_DENSE_LIMIT = 7
        self.HARD_SPARSE_LIMIT = 20

        # Gates that destroy sparsity rapidly
        self.SCRAMBLING_GATES = {'h', 'rx', 'ry', 'qft'}

        # Default mode, will be overwritten by analysis
        self.mode = 'tbd'

    def _estimate_sparsity(self, circuit: QuantumCircuit) -> float:
        """
        Estimates the fraction of non-zero states in the final vector.
        """
        n = circuit.n
        current_active_paths = 1.0
        max_paths = 2.0 ** n

        touched_qubits = set()

        branching_gates = {'h', 'rx', 'ry', 'crx', 'cry', 'qft'}

        for gate in circuit.gates:
            name = gate.name.lower().split('(')[0]

            # Track qubits involved in branching/superposition
            if name in branching_gates:
                current_active_paths *= 2.0
                touched_qubits.update(gate.targets)
                if gate.controls: touched_qubits.update(gate.controls)

            # Can't exceed total Hilbert space (max_paths)
            # Can't exceed subspace of touched qubits (2^len(touched))
            saturation_limit = 2.0 ** len(touched_qubits)
            current_active_paths = min(current_active_paths, max_paths, saturation_limit)

            # If density > 20%, it is Dense.
            if (current_active_paths / max_paths) > 0.2:
                return 0.25

        return current_active_paths / max_paths

    def analyze_mode(self, circuit: QuantumCircuit) -> str:
        """
        Decides simulation mode based on N and Estimated Sparsity.
        """
        n = circuit.n

        if n >= self.HARD_SPARSE_LIMIT:
            self.mode = 'sparse'
            return 'sparse'

        sparsity_score = self._estimate_sparsity(circuit)

        is_dense_enough = sparsity_score > 0.15
        is_small_enough = n < self.HARD_DENSE_LIMIT + 1

        if is_small_enough and is_dense_enough:
            self.mode = 'dense'
        else:
            self.mode = 'sparse'

        return self.mode

    def optimise_circuit(self, circuit: QuantumCircuit, noise_model: Optional[NoiseModel] = None) -> QuantumCircuit:
        """
        Compiles the circuit for execution.
        """
        # Determine mode for this circuit if not already set
        if self.mode == 'tbd':
            self.analyze_mode(circuit)

        # Work on a copy
        optimised_qc = circuit.copy()

        # Apply Noise
        if noise_model is not None:
            optimised_qc = noise_model.apply(optimised_qc)

        # Optimise (Fusion + Reordering)
        if hasattr(optimised_qc, 'optimise'):
            optimised_qc.optimise()
        else:
            optimised_qc.optimise()

        return optimised_qc

    def execute(self, circuit: QuantumCircuit, initial_state: QuantumState = None, shots: int = 1,inplace: bool = False) -> Union[QuantumState, Dict[str, int]]:
        """
        Runs the circuit exactly as provided (NO optimization).

        Args:
            inplace (bool): If True, modifies initial_state directly. Unsafe for general use.
        """
        # Ensure mode is set
        if self.mode == 'tbd':
            self.analyze_mode(circuit)

        current_mode = self.mode

        # Prepare State
        if initial_state is None:
            state = QuantumState(circuit.n, mode=current_mode)
        else:
            if inplace:
                state = initial_state
            else:
                state = initial_state.copy()

            if state.mode != current_mode:
                if current_mode == 'sparse':
                    state.to_sparse()
                else:
                    state.to_dense()

        # Execute
        if shots == 1:
            circuit.run(state, method=current_mode)
            return state
        else:
            return circuit.simulate(state, shots=shots, method=current_mode)

    # Legacy wrapper for backward compatibility if needed.
    def run(self, circuit: QuantumCircuit, initial_state: QuantumState = None, shots: int = 1, optimise: bool = True,
            noise_model: Optional[NoiseModel] = None) -> Union[QuantumState, Dict[str, int]]:
        """
        One-shot execution wrapper. Optimises and runs.
        """
        qc = circuit
        self.mode = 'tbd'

        if optimise or noise_model:
            qc = self.optimise_circuit(circuit, noise_model)

        return self.execute(qc, initial_state, shots=shots)