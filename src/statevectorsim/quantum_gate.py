import numpy as np
from .quantum_state import QuantumState

class QuantumGate:
    """Represents a quantum gate."""
    def __init__(self, matrix: np.ndarray, targets: list[int]):
        self.matrix = matrix
        self.targets = targets  # list of qubits it acts on

    @staticmethod
    def x(target: int):
        """Single Pauli-X (NOT) gate"""
        return QuantumGate(np.array([[0, 1], [1, 0]], dtype=complex), [target])

    @staticmethod
    def y(target: int):
        """Single Pauli-Y gate"""
        return QuantumGate(np.array([[0, -1j], [1j, 0]], dtype=complex), [target])

    @staticmethod
    def z(target: int):
        """Single Pauli-Z gate"""
        return QuantumGate(np.array([[1, 0], [0, -1]], dtype=complex), [target])

    @staticmethod
    def h(target: int):
        """Single Hadamard gate"""
        return QuantumGate((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex), [target])


    @staticmethod
    def i(target: int):
        """Identity gate"""
        return QuantumGate(np.eye(2, dtype=complex), [target])

    @staticmethod
    def cx(control: int, target: int):
        """CNOT gate (2 qubits)"""
        return QuantumGate(np.array([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,0,1],
                                     [0,0,1,0]], dtype=complex),
                           [control, target])

    def apply(self, quantum_state):
        """Apply gate to QuantumState using bitmask"""
        state = quantum_state.state # statevector
        n = quantum_state.n # number of qubits
        k = len(self.targets) # number of targets
        size = 1 << n  # represent 2^n as bit-shift.

        # Precompute bitmasks for target qubits. 1 << t sets t-th bit to 1, all others to 0.
        target_bits = [1 << t for t in self.targets]

        # Iterate through the statevector in blocks
        for i in range(size):

            # Only handle base indices where all target bits are 0). Avoids doing blocks more than once.
            if any(i & b for b in target_bits):
                continue

            # Compute over combinations of k target qubits. 2^k total. Can probably parallelise here.
            indices = []
            for mask in range(1 << k):
                j = i # start from base index

                # iterate over target qubits
                for bit_idx in range(k):

                    # check if target qubit should be set to 1 in this combination
                    if mask & (1 << bit_idx):

                        # bitwise OR to set qubit
                        j |= target_bits[bit_idx]

                # add j to one of 2^k indices the block represents
                indices.append(j)

            # Extract target amplitudes
            block = np.array([state[j] for j in indices])

            # apply gate
            new_block = self.matrix @ block

            # overwrite statevector amplitudes
            for j, val in zip(indices, new_block):
                state[j] = val

        quantum_state.state = state
