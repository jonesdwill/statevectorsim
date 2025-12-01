import numpy as np
from typing import List, Tuple
from .quantum_gate import QuantumGate
from .quantum_circuit import QuantumCircuit


class QuantumChannel(QuantumGate):
    """
    Quantum noise channel. Applies unitary operators based on a probability distribution (Monte Carlo simulation).

    Attributes:
        unitaries (List[np.ndarray]): List of unitary matrices representing possible errors.
        probabilities (List[float]): Probabilities of each unitary. Must sum to 1.0.
        targets (List[int]): Indices of target qubits.
        name (str): Name of the channel.

    """

    def __init__(self, unitaries: List[np.ndarray], probabilities: List[float], targets: List[int], name: str = "Noise"):
        """
        Args:
            unitaries (List[np.ndarray]): List of unitary matrices representing possible errors.
            probabilities (List[float]): Probabilities of each unitary. Must sum to 1.0.
            targets (List[int]): Indices of target qubits.
            name (str): Name of the channel.

        Raises:
            ValueError: If the probabilities do not sum to 1.0 (within numerical tolerance).
        """

        self.unitaries = unitaries
        self.probabilities = probabilities
        self.targets = targets
        self.controls = []
        self.name = name

        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError(f"Noise probabilities must sum to 1.0, got {sum(probabilities)}")

    def apply(self, quantum_state, method: str = 'dense'):
        """
        Applies the noise channel to a quantum state. Randomly selects one unitary based on the assigned probabilities
        and applies it to the state vector.

        Args:
            quantum_state (QuantumState): State to be modified.
            method (str, optional): Simulation backend ('dense' or 'sparse').
        """

        # Select index based on probabilities
        selected_index = np.random.choice(len(self.unitaries), p=self.probabilities)

        # Get unitary matrix
        matrix = self.unitaries[selected_index]

        # If matrix is Identity, do nothing
        if np.allclose(matrix, np.eye(matrix.shape[0])):
            return

        # Create a temporary standard QuantumGate to leverage existing solvers
        temp_gate = QuantumGate(matrix, self.targets, [], name=f"{self.name}_Sample")
        temp_gate.apply(quantum_state, method=method)


    # ------------------------------------------------
    #            Standard Noise Models
    # ------------------------------------------------

    @staticmethod
    def depolarizing(target: int, p: float) -> 'QuantumChannel':
        """
        Single-qubit Depolarizing channel.

        - Prob 1-p: Identity (Success)
        - Prob p/3: X Error (Bit Flip)
        - Prob p/3: Y Error (Bit+Phase Flip)
        - Prob p/3: Z Error (Phase Flip)

        Args:
            target (int): Target qubit index.
            p (float): Probability of depolarisation.

        Returns:
            QuantumChannel: Configured depolarizing channel.
        """

        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        return QuantumChannel(
            unitaries=[I, X, Y, Z],
            probabilities=[1 - p, p / 3, p / 3, p / 3],
            targets=[target],
            name=f"Depolarize({p})"
        )

    @staticmethod
    def bit_flip(target: int, p: float) -> 'QuantumChannel':
        """ Applies X error with probability p. """
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        return QuantumChannel([I, X], [1 - p, p], [target], name=f"BitFlip({p})")

    @staticmethod
    def phase_flip(target: int, p: float) -> 'QuantumChannel':
        """ Applies Z error with probability p. """
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return QuantumChannel([I, Z], [1 - p, p], [target], name=f"PhaseFlip({p})")


class NoiseModel:
    """
    A configuration object for managing error and injecting noise into QuantumCircuits.

    Supports a uniform noise model where error rate is applied to every gate in the circuit.
    """


    def __init__(self, default_error_rate: float = 0.0):
        """
        Args:
            default_error_rate (float): global probability of error for every gate operation.
        """

        self.default_error_rate = default_error_rate

    def apply(self, circuit: 'QuantumCircuit') -> 'QuantumCircuit':
        """
        Returns QuantumCircuit with noise injected after every gate.

        Args:
            circuit (QuantumCircuit): The clean input circuit.

        Returns:
            QuantumCircuit: A new circuit with noise channels injected.
        """
        from .quantum_circuit import QuantumCircuit

        noisy_qc = QuantumCircuit(circuit.n)

        for gate in circuit.gates:
            noisy_qc.add_gate(gate)
            p = self.default_error_rate

            if p > 0:
                # Inject noise on all qubits involved in the gate
                touched_qubits = gate.targets + gate.controls
                for q in touched_qubits:
                    noisy_qc.add_gate(QuantumChannel.depolarizing(q, p))

        return noisy_qc