import numpy as np
import math

class QuantumState:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=complex)
        self.basis_state()

    def basis_state(self, index: int = 0):
        self.state[:] = 0
        self.state[index] = 1.0

    def statevector(self):
        return self.state.copy()

    def get_probabilities(self):
        return np.abs(self.state) ** 2


    def copy(self) -> 'QuantumState':
        """ Copy state for multi-shot runs. """
        new_state = self.__class__(self.n)
        new_state.state = self.state.copy()
        return new_state

    def measure_qubit(self, qubit: int):
        """Measure a single qubit (big-endian) without reshaping or permuting."""
        target_bit = 1 << qubit

        # find indices where qubit is 0 or 1
        indices_0 = np.where((np.arange(len(self.state)) & target_bit) == 0)[0]
        indices_1 = np.where((np.arange(len(self.state)) & target_bit) != 0)[0]

        # compute probabilities
        p0 = np.sum(np.abs(self.state[indices_0]) ** 2)
        p1 = 1 - p0

        # Ensure probabilities are clipped to [0, 1] - reduce floating point errors
        probabilities = np.clip([p0, p1], 0.0, 1.0)
        probabilities /= np.sum(probabilities)
        outcome = np.random.choice([0, 1], p=probabilities)

        # collapse statevector
        if outcome == 0:
            self.state[indices_1] = 0
        else:
            self.state[indices_0] = 0

        # normalize
        self.state /= np.linalg.norm(self.state)

        return outcome


    def measure_all(self):
        """Measure all qubits in computational basis."""

        # probabilities
        probability_vector = np.abs(self.state) ** 2

        # sample one index
        index = np.random.choice(len(self.state), p=probability_vector)

        # convert to bitstring (big-endian)
        outcome = [(index >> i) & 1 for i in reversed(range(self.n))]

        # collapse statevector
        self.state[:] = 0
        self.state[index] = 1.0

        return outcome