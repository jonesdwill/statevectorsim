import numpy as np
import math
from typing import Union, Dict, List
from scipy.sparse import csr_matrix, lil_matrix

class QuantumState:
    def __init__(self, n_qubits: int, mode: str = 'sparse', tolerance: float = 1e-12):
        self.n = n_qubits
        self.dim = 2 ** n_qubits

        # --- Dual State Storage ---
        # Dense storage (NumPy array)
        self.state = np.zeros(self.dim, dtype=complex)
        # Sparse storage (SciPy sparse matrices)
        self.sparse_state: csr_matrix = csr_matrix((1, self.dim), dtype=complex)

        # Define a tolerance for sparse cleanup
        self.TOLERANCE = tolerance

        # Mode Tracking
        self.mode = mode if mode in ['dense', 'sparse'] else 'sparse'

        self.basis_state()

    def basis_state(self, index: int = 0):
        """Initializes the state to the specified basis state |index>."""

        # Set the dense state (for consistency and small states)
        self.state[:] = 0
        self.state[index] = 1.0 + 0j

        # Create a sparse LIL matrix
        lil = lil_matrix((1, self.dim), dtype=complex)
        lil[0, index] = 1.0 + 0j

        # Convert to CSR for faster multiplication
        self.sparse_state = lil.tocsr()

        # Ensure active mode is correct after initialization
        self.mode = 'dense' if self.mode == 'dense' else 'sparse'

    def statevector(self) -> np.ndarray:
        """Returns current state vector as a dense NumPy array."""
        self.to_dense()
        return self.state.copy()

    def get_probabilities(self) -> np.ndarray:
        """Calculates and returns the probability vector."""
        self.to_dense()
        return np.abs(self.state) ** 2

    def copy(self) -> 'QuantumState':
        """ Copy state for multi-shot runs. """

        new_state = self.__class__(self.n, mode=self.mode)

        if self.mode == 'dense':
            new_state.state = self.state.copy()
            new_state.sparse_state = self.to_sparse(self.state)
        else:
            # Copy and pass the CSR matrix directly
            new_state.sparse_state = self.sparse_state.copy()
            new_state.state = self.to_dense(self.sparse_state)

        return new_state

    def clean_sparse(self):
        """Removes all amplitudes from the sparse state that are below tolerance."""
        # SciPy's CSR format handles cleanup
        self.sparse_state.eliminate_zeros()

    # -------------------------------------
    #       Storage Rep Conversion
    # -------------------------------------

    def to_dense(self, source: Union[csr_matrix, None] = None) -> np.ndarray:
        """
        Converts active sparse state (self.sparse_state) to dense NumPy array (self.state).
        Source can be any CSR matrix (though usually self.sparse_state).
        """
        if source is None:
            source = self.sparse_state
            self.state = source.toarray().flatten()

            self.mode = 'dense'
            return self.state
        else:
            return source.toarray().flatten()

    def to_sparse(self, source: Union[np.ndarray, None] = None) -> csr_matrix:
        """
        Converts active dense state (self.state) to sparse CSR matrix (self.sparse_state).
        Source can be any NumPy array (though commonly self.state).
        """
        if source is None:
            source = self.state

            self.sparse_state = csr_matrix(source)
            self.sparse_state.eliminate_zeros()

            self.mode = 'sparse'
            return self.sparse_state
        else:
            sparse_matrix = csr_matrix(source)
            sparse_matrix.eliminate_zeros()
            return sparse_matrix

    # -------------------------------------
    #            Measurement
    # -------------------------------------

    def measure_qubit(self, qubit: int):
        """
        Measure a single qubit. Requires temporary conversion to dense mode.
        """

        # Temporarily switch to dense mode for NumPy-based measurement
        original_mode = self.mode
        self.to_dense()

        state = self.state # use the dense state
        target_bit = 1 << qubit

        # find indices where qubit is 0 or 1
        indices_0 = np.where((np.arange(len(state)) & target_bit) == 0)[0]
        indices_1 = np.where((np.arange(len(state)) & target_bit) != 0)[0]

        # compute probabilities
        p0 = np.sum(np.abs(state[indices_0]) ** 2)
        p1 = 1 - p0

        # Ensure probabilities are clipped to [0, 1] - reduce floating point errors
        probabilities = np.clip([p0, p1], 0.0, 1.0)
        probabilities /= np.sum(probabilities)
        outcome = np.random.choice([0, 1], p=probabilities)

        # collapse statevector
        if outcome == 0:
            state[indices_1] = 0
        else:
            state[indices_0] = 0

        # normalize
        state /= np.linalg.norm(state)
        self.state = state # update the dense state

        # Convert back to the original mode if it was sparse
        if original_mode == 'sparse':
            self.to_sparse()

        # Set final mode
        self.mode = original_mode

        return outcome

    def measure_all(self):
        """
        Measure all qubits in computational basis. Requires temporary conversion to dense mode.
        """

        # Temporarily switch to dense mode for NumPy-based measurement
        original_mode = self.mode
        self.to_dense()

        state = self.state # use the dense state

        # probabilities
        probability_vector = np.abs(state) ** 2

        # sample one index
        index = np.random.choice(len(state), p=probability_vector)

        # convert to bitstring (big-endian)
        outcome = [(index >> i) & 1 for i in reversed(range(self.n))]

        # collapse statevector
        state[:] = 0
        state[index] = 1.0
        self.state = state # update the dense state

        # Convert back to the original mode if it was sparse
        if original_mode == 'sparse':
            self.to_sparse()

        # Set final mode
        self.mode = original_mode

        return outcome