import numpy as np
import math
from typing import Union
from scipy.sparse import csr_matrix, lil_matrix


class QuantumState:
    def __init__(self, n_qubits: int, mode: str = 'sparse', tolerance: float = 1e-12):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.TOLERANCE = tolerance

        # --- Dual State Storage ---
        # Rename internal dense storage to _state
        self._state = np.zeros(self.dim, dtype=complex)

        # Sparse storage
        self.sparse_state: csr_matrix = csr_matrix((1, self.dim), dtype=complex)

        # Mode Tracking
        self.mode = mode if mode in ['dense', 'sparse'] else 'sparse'

        # Initialize
        self.basis_state(0)

    @property
    def state(self) -> np.ndarray:
        """
        Property that ensures the dense state is up-to-date before returning it.
        """
        if self.mode == 'sparse':
            self.to_dense()
        return self._state

    @state.setter
    def state(self, value: np.ndarray):
        """
        Setter to update the dense state manually.
        """
        self._state = value

    def basis_state(self, index: int = 0):
        """Initializes the state to the specified basis state |index>."""
        # Set dense
        self._state[:] = 0
        self._state[index] = 1.0 + 0j

        # Set sparse
        lil = lil_matrix((1, self.dim), dtype=complex)
        lil[0, index] = 1.0 + 0j
        self.sparse_state = lil.tocsr()

        # Default to sparse mode for fresh states unless specified otherwise
        # (Your original toggle logic was a bit unpredictable, setting to sparse is safer)
        if self.mode == 'dense':
            self.mode = 'dense'
        else:
            self.mode = 'sparse'

    def statevector(self) -> np.ndarray:
        """Returns current state vector as a dense NumPy array."""
        # The property access 'self.state' will now trigger to_dense() automatically
        return self.state.copy()

    def get_probabilities(self) -> np.ndarray:
        """Calculates and returns the probability vector."""
        return np.abs(self.state) ** 2

    def copy(self) -> 'QuantumState':
        """ Copy state for multi-shot runs. """
        new_state = self.__class__(self.n, mode=self.mode)

        if self.mode == 'dense':
            new_state.state = self._state.copy()
            new_state.sparse_state = self.to_sparse(self._state)
        else:
            new_state.sparse_state = self.sparse_state.copy()
            # We don't force dense conversion here to keep copy fast
            # But we update the backing field just in case
            if self.mode == 'dense':
                new_state.state = self._state.copy()

        return new_state

    def clean_sparse(self, tolerance: float = 1e-10):
        """
        Removes all amplitudes from the sparse state that are below tolerance.
        Handles complex magnitudes correctly.
        """
        # Identify noise: values where magnitude is less than tolerance
        mask = np.abs(self.sparse_state.data) < tolerance

        # Set those values to exact zero
        self.sparse_state.data[mask] = 0.0 + 0.0j

        # Remove the explicit zeros from the sparsity structure
        self.sparse_state.eliminate_zeros()

    # -------------------------------------
    #       Storage Rep Conversion
    # -------------------------------------

    def to_dense(self, source: Union[csr_matrix, None] = None) -> np.ndarray:
        """
        Converts active sparse state to dense NumPy array.
        """
        if source is None:
            source = self.sparse_state
            # Update backing field directly
            self._state = source.toarray().flatten()
            self.mode = 'dense'
            return self._state
        else:
            return source.toarray().flatten()

    def to_sparse(self, source: Union[np.ndarray, None] = None) -> csr_matrix:
        """
        Converts active dense state to sparse CSR matrix.
        """
        if source is None:
            # Use backing field
            source = self._state
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
        # Temporarily switch to dense mode via property access
        # caching original mode to restore later if desired
        original_mode = self.mode

        state = self.state  # Triggers sync if sparse
        target_bit = 1 << qubit

        indices_0 = np.where((np.arange(len(state)) & target_bit) == 0)[0]
        indices_1 = np.where((np.arange(len(state)) & target_bit) != 0)[0]

        p0 = np.sum(np.abs(state[indices_0]) ** 2)
        p1 = 1 - p0
        probabilities = np.clip([p0, p1], 0.0, 1.0)
        probabilities /= np.sum(probabilities)

        outcome = np.random.choice([0, 1], p=probabilities)

        if outcome == 0:
            state[indices_1] = 0
        else:
            state[indices_0] = 0

        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm

        self._state = state

        # Restore sparse rep if needed
        if original_mode == 'sparse':
            self.to_sparse()

        return outcome

    def measure_all(self):
        original_mode = self.mode

        state = self.state  # Triggers sync
        probability_vector = np.abs(state) ** 2

        # Normalize just in case
        probability_vector /= np.sum(probability_vector)

        index = np.random.choice(len(state), p=probability_vector)
        outcome = [(index >> i) & 1 for i in reversed(range(self.n))]

        # Collapse
        self._state[:] = 0
        self._state[index] = 1.0

        if original_mode == 'sparse':
            self.to_sparse()

        return outcome