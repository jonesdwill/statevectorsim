import numpy as np
from typing import Dict, Union, List, Optional
from .quantum_circuit import QuantumCircuit
from .quantum_state import QuantumState
from .quantum_noise import NoiseModel


class QuantumBackend:
    """
    Intelligent simulation dispatcher separating circuit compilation from execution logic.

    Backend analyzes circuit's structure and available system resources to automatically select the optimal simulation strategy:
    1. 'Dense' Mode (NumPy): Best for small (<20 qubit) circuits or highly entangled,
       superposition-heavy states where the statevector is dense.
    2. 'Sparse' Mode (SciPy): Best for large (>20 qubit) circuits or those that remain
       computationally sparse (e.g., GHZ states, reversible logic).

    Attributes:
        HARD_DENSE_LIMIT (int): Heuristic qubit count below which dense is generally preferred.
        HARD_SPARSE_LIMIT (int): Heuristic qubit count above which sparse is mandatory to avoid memory issues.
        SCRAMBLING_GATES (set): Set of gates known to rapidly destroy sparsity.
        mode (str): The currently selected execution mode ('dense', 'sparse', or 'tbd').
    """

    def __init__(self):
        """Initialize the QuantumBackend with default heuristics."""
        # Heuristic thresholds
        self.HARD_DENSE_LIMIT = 7
        self.HARD_SPARSE_LIMIT = 20

        # Gates that destroy sparsity rapidly
        self.SCRAMBLING_GATES = {'h', 'rx', 'ry', 'qft'}

        # Default mode, will be overwritten by analysis
        self.mode = 'tbd'

    def _estimate_sparsity(self, circuit: QuantumCircuit) -> float:
        """
        Estimates the fraction of non-zero amplitudes in the final statevector.

        Args:
            circuit (QuantumCircuit): The circuit to analyze.

        Returns:
            float: Estimated sparsity ratio (active_paths / total_hilbert_space).
                   1.0 means fully dense, 0.0 means fully sparse.
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
        Analyzes the circuit to determine the optimal simulation mode.

        Logic:
        1. If n >= 20, force 'sparse' (NumPy arrays of this size exceed typical RAM).
        2. Calculate estimated sparsity score.
        3. If circuit is small (< 8 qubits) and dense enough, use 'dense' (NumPy is faster for small overhead).
        4. Otherwise, default to 'sparse'.

        Args:
            circuit (QuantumCircuit): The circuit to analyze.

        Returns:
            str: The selected mode ('dense' or 'sparse').
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

        Args:
            circuit (QuantumCircuit): The input circuit.
            noise_model (NoiseModel, optional): Noise model to apply. Defaults to None.

        Returns:
            QuantumCircuit: The optimized (and potentially noisy) circuit ready for execution.
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
        Executes the circuit directly without optimising.

        Low-level execution primitive. Handles state initialization, mode switching (converting state to dense/sparse if needed), and running the gates only.

        Args:
            circuit (QuantumCircuit): The circuit to run.
            initial_state (QuantumState, optional): Custom starting state. If None, starts at |0...0>.
            shots (int, optional): Number of execution shots.
                                   - If 1: Returns the final quantum state vector (QuantumState).
                                   - If >1: Returns measurement counts (Dict[str, int]).
            inplace (bool, optional): If True, modifies the provided initial_state in place. Defaults to False.

        Returns:
            Union[QuantumState, Dict[str, int]]: The final state (if shots=1) or measurement results (if shots > 1).
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

    # Legacy wrapper.
    def run(self, circuit: QuantumCircuit, initial_state: QuantumState = None, shots: int = 1, optimise: bool = True,
            noise_model: Optional[NoiseModel] = None) -> Union[QuantumState, Dict[str, int]]:
        """
        High-level convenience wrapper for running a simulation in one step.

        Combines `optimise_circuit` and `execute`.

        Args:
            circuit (QuantumCircuit): The circuit to run.
            initial_state (QuantumState, optional): Custom starting state.
            shots (int, optional): Number of shots. Defaults to 1.
            optimise (bool, optional): Whether to run the optimizer. Defaults to True.
            noise_model (NoiseModel, optional): Noise model to apply. Defaults to None.

        Returns:
            Union[QuantumState, Dict[str, int]]: The simulation result.
        """
        qc = circuit
        self.mode = 'tbd'

        if optimise or noise_model:
            qc = self.optimise_circuit(circuit, noise_model)

        return self.execute(qc, initial_state, shots=shots)