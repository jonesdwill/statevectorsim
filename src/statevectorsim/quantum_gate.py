import numpy as np
import math
from typing import Union, List, Dict
from .quantum_state import QuantumState
from scipy.sparse import csr_matrix, identity, block_diag, kron, isspmatrix_csr, lil_matrix


def _generate_permutation_matrix(n: int, current_order: List[int], target_order: List[int]) -> csr_matrix:
    """
    Generates a sparse permutation matrix to reorder qubits in a statevector.

    Args:
        n (int): Total number of qubits.
        current_order (List[int]): The current qubit ordering (usually [0, 1, ..., n-1]).
        target_order (List[int]): The desired qubit ordering.

    Returns:
        csr_matrix: The 2^n x 2^n permutation matrix.
    """

    if current_order == target_order:
        return identity(2 ** n, dtype=complex, format='csr')

    dim = 2 ** n
    lil = lil_matrix((dim, dim), dtype=complex)

    for old_idx in range(dim):
        new_idx = 0

        # Iterate over the sequential positions in the new basis (j = 0, 1, ..., N-1)
        for j in range(n):
            # The qubit that occupies the j-th bit in the new index is target_order[j]
            q_orig_idx = target_order[j]

            # Check the state of the original qubit (bit q_orig_idx in old_idx)
            if (old_idx >> q_orig_idx) & 1:
                # If the qubit is |1>, set the bit corresponding to the new position 'j'
                new_idx |= (1 << j)

        lil[new_idx, old_idx] = 1.0

    return lil.tocsr()


class QuantumGate:
    """
    Represents a Unitary Quantum Gate.

    Handles gate matrices and their application. Supports application by:
    1. Tensor Slicing (Dense, small k): Fast for single-qubit gates on dense states.
    2. Bitmasking (Dense, controls): Handles controlled gates by iterating indices.
    3. Sparse Multiplication (Sparse): Efficient for large N with sparse states.

    Attributes:
        matrix (np.ndarray): The 2^k x 2^k unitary matrix of the gate.
        targets (list[int]): Indices of the target qubits.
        controls (list[int]): Indices of the control qubits.
        name (str): display name of the gate.
    """

    def __init__(self, matrix: np.ndarray, targets: list[int], controls: list[int] = None, name: str = 'CustomGate'):
        """
          Initialize a QuantumGate.

          Args:
              matrix (np.ndarray): unitary matrix (2^k x 2^k).
              targets (list[int]): The target qubit indices.
              controls (list[int], optional): The control qubit indices. Defaults to None.
              name (str, optional): Name for identification. Defaults to 'CustomGate'.

          Raises:
              ValueError: If matrix dimensions don't match the number of targets or if targets/controls overlap.
          """

        # robust check: matrix dimension is only checked against targets
        expected_dim = 2 ** len(targets)
        if matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Matrix shape {matrix.shape} does not match "
                f"expected size ({expected_dim}, {expected_dim}) "
                f"for {len(targets)} targets."
            )
        # gate assignment
        self.name = name
        self.matrix = matrix
        self.targets = [int(t) for t in targets]
        self.controls = [int(c) for c in (controls or [])]

        # Check for overlaps
        if set(targets) & set(self.controls):
            raise ValueError("Target and control qubits must be distinct.")

    def __str__(self):
        """String representation of the QuantumGate."""
        n_qubits = len(self.targets) + len(self.controls)
        targets_str = ", ".join(map(str, self.targets))
        controls_str = f" Controls: [{', '.join(map(str, self.controls))}]" if self.controls else ""
        return f"{n_qubits}-Qubit {self.name} Gate on Targets [{targets_str}]{controls_str}"

    # -------------------------------------
    #           Helper Methods
    # -------------------------------------

    @staticmethod
    def _create_single_gates(targets: Union[int, List[int]], matrix: np.ndarray, name: str = 'CustomSingleGate') -> List['QuantumGate']:
        """Helper to apply single-qubit gate to one or multiple qubits."""
        if isinstance(targets, int):
            targets = [targets]

        # Single gates have no controls
        return [QuantumGate(matrix, [t], controls=[], name=name) for t in targets]

    @staticmethod
    def _id(n_qubits: int) -> np.ndarray:
        """Returns the 2^n x 2^n identity matrix."""
        dim = 2 ** n_qubits
        return np.identity(dim, dtype=complex)

    def _build_sparse_full_matrix(self, n_qubits: int) -> csr_matrix:
        """
        Constructs the full 2^N x 2^N sparse matrix for the gate U.
        Handles target permutations and control logic.
        """
        N = n_qubits

        # Qubit Order and Permutation
        active_qubits = set(self.targets) | set(self.controls)
        non_active_qubits = sorted([i for i in range(N) if i not in active_qubits])

        # Targets (LSBs of the active block)
        reversed_targets = list(reversed(self.targets))

        # Controls (MSBs of the active block)
        reversed_controls = list(reversed(self.controls))

        # [Targets(LSB..MSB), Controls(LSB..MSB), NonActive]
        new_order = reversed_targets + reversed_controls + non_active_qubits

        original_order = list(range(N))

        # Get the Permutation Matrix P
        P = _generate_permutation_matrix(N, original_order, new_order)
        P_T = P.transpose().tocsr()

        # Build Active Subspace Gate
        if not self.controls:
            G_active = csr_matrix(self.matrix)
        else:
            U_matrix_csr = csr_matrix(self.matrix)
            dim_k = 2 ** len(self.targets)
            dim_control = 2 ** len(self.controls)

            # Size = (2^|controls| - 1) * 2^|targets|
            dim_i_off = (dim_control - 1) * dim_k
            I_control_off = identity(dim_i_off, dtype=complex, format='csr')

            # Controls are now the MSBs of this block.
            G_active = block_diag([I_control_off, U_matrix_csr], format='csr')

        # Embed G_active into the Full Permuted Space ('U')
        I_non_active = identity(2 ** len(non_active_qubits), dtype=complex, format='csr')

        # kron(A, B) -> A is MSB, B is LSB.
        U_prime = kron(I_non_active, G_active, format='csr')

        # Inverse Permutation
        U_full = P_T.dot(U_prime.dot(P))

        return U_full.tocsr()

    # -------------------------------------------
    #    Apply Gate Logic (Speed-up bottleneck)
    # -------------------------------------------

    def _apply_tensor(self, state: np.ndarray, n: int, k: int) -> np.ndarray:
        """
        Multi-qubit gate application using NumPy's efficient tensor reshape and permutation.

        Reshapes the statevector into a tensor of shape (2, 2, ..., 2), permutes the axes so that the target qubits are at the start,
        and applies the gate matrix. Avoids creating the full 2^N x 2^N unitary matrix.

        Args:
            state (np.ndarray): dense statevector (size 2^n).
            n (int): Total number of qubits.
            k (int): Number of target qubits for this gate.

        Returns:
            np.ndarray: The new statevector after applying the gate.
        """

        gate_matrix = self.matrix

        # reshape state vector into an N-dimensional tensor. (q_{n-1}, q_{n-2}, ..., q_0)
        tensor_state = state.reshape([2] * n)

        # map qubit indices to tensor axes. 'i' maps to tensor axis 'n - 1 - i'.
        target_axes = [n - 1 - q_idx for q_idx in self.targets]

        # define new axis: target qubit axes first, then the rest
        all_axes = list(range(n))
        axes_permuted = target_axes + [a for a in all_axes if a not in target_axes]

        # apply forward permutation to bring targets to the front (axes 0 to k-1)
        tensor_state = np.transpose(tensor_state, axes_permuted)

        # reshape state tensor for matrix multiplication by gate (2^k, 2^(n-k))
        dims_other = tensor_state.shape[k:]
        tensor_state = tensor_state.reshape(2 ** k, -1)

        # apply the gate matrix
        tensor_state = gate_matrix @ tensor_state

        # reshape back to original tensor shape
        tensor_state = tensor_state.reshape([2] * k + list(dims_other))

        # inverse qubit permutation
        axes_original = np.argsort(axes_permuted)
        tensor_state = np.transpose(tensor_state, axes_original)

        # flatten and return
        return tensor_state.reshape(-1)

    def _apply_bitmask(self, state: np.ndarray, n: int, k: int) -> np.ndarray:
        """
        Multi-qubit gate application using generalized index-based iteration.

        To be used when controls are present. Iterates through the
        statevector indices, identifies blocks where the target qubits change
        and other targets are fixed, checks the control condition, and applies
        matrix multiplication only to valid blocks.

        Args:
            state (np.ndarray): dense statevector.
            n (int): Total number of qubits.
            k (int): Number of target qubits.

        Returns:
            np.ndarray: The updated statevector.
        """

        gate_matrix = self.matrix
        dim_k = 2 ** k
        new_state = state.copy()

        # Sort targets to ensure the statevector block's internal order matches the gate matrix's order.
        sorted_targets = sorted(self.targets)

        # Pre-calculate the index offsets (for the target bits)
        index_offsets = np.zeros(dim_k, dtype=int)
        for m in range(dim_k):
            offset = 0
            for j, t in enumerate(sorted_targets):
                if (m >> j) & 1:
                    offset |= (1 << t)
            index_offsets[m] = offset

        # Mask of all target qubits (for finding block starts)
        target_mask = sum(1 << t for t in self.targets)

        # Mask of all control qubits (for checking condition)
        control_mask = sum(1 << c for c in self.controls)

        # The value the control bits must be set to (all ones)
        control_check_value = control_mask

        size = 1 << n

        for i in range(size):
            # Identify block start: 'i' where all targets are |0>
            if i & target_mask:
                continue

            # Control check: Skip block if controls are not all |1>
            if self.controls and (i & control_mask) != control_check_value:
                continue

            # extract the 2^k amplitudes from the statevector into a temporary block
            block = np.zeros(dim_k, dtype=complex)
            for m in range(dim_k):
                state_idx = i | index_offsets[m]
                block[m] = state[state_idx]

            # Apply gate matrix
            new_block = gate_matrix @ block

            # Write new block back to the statevector
            for m in range(dim_k):
                state_idx = i | index_offsets[m]
                new_state[state_idx] = new_block[m]

        return new_state

    def _apply_sparse(self, quantum_state: 'QuantumState'):
        """
        Apply gate directly to the non-zero elements of a sparse state vector (CSR).

        Instead of constructing a massive 2^N x 2^N gate matrix, this method
        extracts the active indices from the state, performs small matrix
        multiplications, and reconstructs the sparse result.

        Args:
            quantum_state (QuantumState): The state object (must be in 'sparse' mode).
        """

        # Get current sparse state data (CSR format)
        state_csr = quantum_state.sparse_state
        indices = state_csr.indices
        data = state_csr.data

        if len(indices) == 0:
            return

        n_qubits = quantum_state.n

        # identify active indices
        if self.controls:
            control_mask = 0
            for c in self.controls:
                control_mask |= (1 << c)

            # Check which indices have all control bits set
            # (idx & control_mask) == control_mask
            active_mask = (indices & control_mask) == control_mask
        else:
            # all indices are active if no controls
            active_mask = np.ones(len(indices), dtype=bool)

        # Split indices into active and passive
        active_indices = indices[active_mask]
        active_data = data[active_mask]

        passive_indices = indices[~active_mask]
        passive_data = data[~active_mask]

        if len(active_indices) == 0:
            return

        # apply logic to active indices
        target_bits = np.array(self.targets, dtype=int)

        # create bit masks for targets
        target_mask = 0
        for t in self.targets:
            target_mask |= (1 << t)

        # local index (0..2^k-1) for the gate from the global index
        local_indices = np.zeros(len(active_indices), dtype=int)
        for i, t in enumerate(self.targets):
            # Extract bit 't', shift it to position 'i'
            bit_val = (active_indices >> t) & 1
            local_indices |= (bit_val << i)

        # base index is the global index with target bits zeroed out
        base_indices = active_indices & (~target_mask)

        # Perform the small matrix multiplication
        gate_matrix = self.matrix
        dim_gate = gate_matrix.shape[0]

        new_rows = []
        new_cols = []
        new_vals = []

        # For each output row of the gate matrix (possible outcome state)
        for row_idx in range(dim_gate):
            # Get the matrix row coefficients for the specific local indices
            coeffs = gate_matrix[row_idx, local_indices]

            # Keep only non-zero contributions
            nz_mask = (coeffs != 0)
            if not np.any(nz_mask):
                continue

            valid_coeffs = coeffs[nz_mask]
            valid_data = active_data[nz_mask]
            valid_base = base_indices[nz_mask]

            # Calculate the new global indices
            added_bits = 0
            for i, t in enumerate(self.targets):
                if (row_idx >> i) & 1:
                    added_bits |= (1 << t)

            final_indices = valid_base | added_bits

            # Append data for COO construction
            new_cols.append(final_indices)
            new_vals.append(valid_data * valid_coeffs)

            # Row index is always 0 for statevector
            new_rows.append(np.zeros(len(final_indices), dtype=int))

        # Reconstruct the State
        if new_cols:
            active_cols = np.concatenate(new_cols)
            active_vals = np.concatenate(new_vals)
            active_rows = np.concatenate(new_rows)
        else:
            active_cols = np.array([], dtype=int)
            active_vals = np.array([], dtype=complex)
            active_rows = np.array([], dtype=int)

        # Add passive data
        final_cols = np.concatenate([passive_indices, active_cols])
        final_vals = np.concatenate([passive_data, active_vals])
        final_rows = np.concatenate([np.zeros(len(passive_indices), dtype=int), active_rows])

        # Create new CSR matrix
        from scipy.sparse import coo_matrix
        new_coo = coo_matrix((final_vals, (final_rows, final_cols)), shape=(1, quantum_state.dim), dtype=complex)

        # Update state
        quantum_state.sparse_state = new_coo.tocsr()
        quantum_state.clean_sparse()

    def apply(self, quantum_state, method: str = 'dense'):
        """
        Apply the gate to a QuantumState using the specified method.

        Main dispatch method. It checks the state's mode and requested method, performs conversions if necessary,
        and routes to _apply_tensor, _apply_bitmask, or _apply_sparse.

        Args:
            quantum_state (QuantumState): state to apply the gate to.
            method (str, optional): 'dense' or 'sparse'. Defaults to 'dense'.
        """

        n = quantum_state.n
        k = len(self.targets)
        original_mode = quantum_state.mode

        # Check if targets/controls are valid (within [0, n-1])
        all_qubits = self.targets + self.controls
        if not all(0 <= t < n for t in all_qubits):
            raise ValueError(
                f"Gate qubits {all_qubits} are out of bounds "
                f"for an {n}-qubit state."
            )

        state = None

        # --- Handle State Mode Conversion ---
        if original_mode == 'sparse':
            if method == 'dense':
                quantum_state.to_dense()
                state = quantum_state.state.copy()
            elif method == 'sparse':
                self._apply_sparse(quantum_state)
                quantum_state.mode = 'sparse'
                return
            else:
                raise ValueError(f"Unknown method '{method}'. Choose 'dense' or 'sparse'.")

        elif original_mode == 'dense':
            if method == 'sparse':
                quantum_state.to_sparse()
                self._apply_sparse(quantum_state)
                return
            state = quantum_state.state.copy()


        # --- Internal Dense Routing ---
        if self.controls:
            # Controlled gates (like CRp in QFT) require index-based conditional logic
            internal_method = '_apply_bitmask'
        elif k == 1:
            # Optimization: Single-qubit gates are often simplest/fastest with tensor method
            internal_method = '_apply_tensor'
        else:
            # Multi-target non-controlled gates (e.g., SWAP) also use tensor method
            internal_method = '_apply_tensor'

        if internal_method == '_apply_tensor':
            new_state = self._apply_tensor(state, n, k)
        else:  # internal_method == '_apply_bitmask'
            new_state = self._apply_bitmask(state, n, k)


        # --- Update and Restore
        # Update State
        quantum_state.state = new_state

        # Restore original mode if it was sparse
        if original_mode == 'sparse':
            quantum_state.to_sparse()

    # -------------------------------------
    #    Standard Gates (X, Y, Z, H, I)
    # -------------------------------------

    @staticmethod
    def x(targets: Union[int, List[int]]):
        """Single or multiple Pauli-X gates."""
        matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, 'x')

    @staticmethod
    def y(targets: Union[int, List[int]]):
        """Single or multiple Pauli-Y gates."""
        matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, 'y')

    @staticmethod
    def z(targets: Union[int, List[int]]):
        """Single or multiple Pauli-Z gates."""
        matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, 'z')

    @staticmethod
    def h(targets: Union[int, List[int]]):
        """Single or multiple Hadamard gates."""
        matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, 'h')

    @staticmethod
    def i(targets: Union[int, List[int]]):
        """Identity gate"""
        matrix = np.eye(2, dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, 'i')

    # -------------------------------------
    #     Phase Gates (S, T, Sdag, Tdag)
    # -------------------------------------

    @staticmethod
    def s(targets: Union[int, List[int]]):
        """Phase gate (S-gate, sqrt(Z)). Equivalent to Rz(pi/2)."""
        # Matrix: [[1, 0], [0, i]]
        matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, 's')

    @staticmethod
    def sdg(targets: Union[int, List[int]]):
        """Inverse Phase gate (S-dagger). Equivalent to Rz(-pi/2)."""
        # Matrix: [[1, 0], [0, -i]]
        matrix = np.array([[1, 0], [0, -1j]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, 'sdg')

    @staticmethod
    def t(targets: Union[int, List[int]]):
        """T-gate. Equivalent to Rz(pi/4)."""
        # Matrix: [[1, 0], [0, exp(i*pi/4)]]
        phase = np.exp(1j * math.pi / 4)
        matrix = np.array([[1, 0], [0, phase]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, 't')

    @staticmethod
    def tdg(targets: Union[int, List[int]]):
        """Inverse T-gate (T-dagger). Equivalent to Rz(-pi/4)."""
        # Matrix: [[1, 0], [0, exp(-i*pi/4)]]
        phase = np.exp(-1j * math.pi / 4)
        matrix = np.array([[1, 0], [0, phase]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, 'tdg')

    # -------------------------------------
    #      Rotation Gates (RX, RY, RZ)
    # -------------------------------------

    @staticmethod
    def rx(targets: Union[int, List[int]], theta: float):
        """Single or multiple Pauli-X rotation gates. Rx(theta)"""
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)
        matrix = np.array([[c, -1j * s],
                           [-1j * s, c]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, f'rx({theta})')

    @staticmethod
    def ry(targets: Union[int, List[int]], theta: float):
        """Single or multiple Pauli-Y rotation gates. Ry(theta)"""
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)
        matrix = np.array([[c, -s],
                           [s, c]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, f'ry({theta})')

    @staticmethod
    def rz(targets: Union[int, List[int]], theta: float):
        """Single or multiple Pauli-Z rotation gates. Rz(theta)"""
        half_theta = theta / 2.0
        matrix = np.array([[np.exp(-1j * half_theta), 0],
                           [0, np.exp(1j * half_theta)]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix, f'rz({theta})')


    # -------------------------------------
    #  Controlled Gates (CX, CY, CZ, SWAP)
    # -------------------------------------

    @staticmethod
    def cx(control: int, target: int):
        """CNOT gate (2 qubits) - Applies Pauli-X to target if control is |1>."""
        x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        return QuantumGate(x_matrix, targets=[target], controls=[control], name='cx')

    @staticmethod
    def cy(control: int, target: int):
        """Controlled-Y gate (2 qubits) - Applies Pauli-Y to target if control is |1>."""
        y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return QuantumGate(y_matrix, targets=[target], controls=[control], name='cy')

    @staticmethod
    def cz(control: int, target: int):
        """Controlled-Z gate (2 qubits) - Applies Pauli-Z to target if control is |1>."""
        z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        return QuantumGate(z_matrix, targets=[target], controls=[control], name='cz')

    @staticmethod
    def swap(q1: int, q2: int):
        """SWAP gate (2 qubits). A standard 2-qubit gate with NO controls."""
        matrix = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]], dtype=complex)
        return QuantumGate(matrix, targets=[q1, q2], name='swap')

    @staticmethod
    def cu(control: int, target_qubits: list[int], unitary_matrix: np.ndarray, k: int = 1):
        """ Creates a Controlled-U^k gate. """

        m_qubits = len(target_qubits)
        if unitary_matrix.shape != (2 ** m_qubits, 2 ** m_qubits):
            raise ValueError(
                f"Unitary matrix size must be 2^m x 2^m where m is the number of target qubits ({m_qubits})."
            )

        U_k = np.linalg.matrix_power(unitary_matrix, k)

        return QuantumGate(U_k, targets=target_qubits, controls=[control], name='cu')

    # ---------------------------------------------------
    #    Controlled Rotation Gates (CRX, CRY, CRZ, CRP)
    # ---------------------------------------------------

    @staticmethod
    def crx(control: int, target: int, theta: float) -> 'QuantumGate':
        """Controlled-Rx gate."""
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)
        # Rx(theta) matrix
        matrix = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        return QuantumGate(matrix, targets=[target], controls=[control], name=f'crx({theta})')

    @staticmethod
    def cry(control: int, target: int, theta: float) -> 'QuantumGate':
        """Controlled-Ry gate."""
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)
        # Ry(theta) matrix
        matrix = np.array([[c, -s], [s, c]], dtype=complex)
        return QuantumGate(matrix, targets=[target], controls=[control], name=f'cry({theta})')

    @staticmethod
    def crz(control: int, target: int, theta: float) -> 'QuantumGate':
        """Controlled-Rz gate."""
        half_theta = theta / 2.0
        e_neg = np.exp(-1j * half_theta)
        e_pos = np.exp(1j * half_theta)
        # Rz(theta) matrix
        matrix = np.array([[e_neg, 0], [0, e_pos]], dtype=complex)
        return QuantumGate(matrix, targets=[target], controls=[control], name=f'crz({theta})')

    @staticmethod
    def crp(control: int, target: int, theta: float):
        """ Controlled Phase Gate (CRP or CP(theta)). """
        phase = np.exp(1j * theta)
        # P(theta) matrix
        matrix = np.array([[1, 0], [0, phase]], dtype=complex)
        return QuantumGate(matrix, targets=[target], controls=[control], name=f'crp({theta})')

    # -----------------------------------------
    #   Multi-Controlled Gates (MCX, MCY, MCZ)
    # -----------------------------------------

    @staticmethod
    def mcx(controls: List[int], target: int):
        """ Multi-Controlled X (MCX) or N-Controlled NOT gate. """
        if not controls:
            raise ValueError("MCX requires at least one control qubit.")
        x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # 2x2 unitary
        n_qubits = len(controls) + 1
        return QuantumGate(x_matrix, targets=[target], controls=controls, name=f'{n_qubits}-control mcx')

    @staticmethod
    def mcy(controls: List[int], target: int):
        """ Multi-Controlled Y (MCY) gate. """
        if not controls:
            raise ValueError("MCY requires at least one control qubit.")
        y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)  # 2x2 unitary
        n_qubits = len(controls) + 1
        return QuantumGate(y_matrix, targets=[target], controls=controls, name=f'{n_qubits}-control mcy')

    @staticmethod
    def mcz(controls: List[int], target: int):
        """ Multi-Controlled Z (MCZ) gate. """
        if not controls:
            raise ValueError("MCZ requires at least one control qubit.")
        z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)  # 2x2 unitary
        n_qubits = len(controls) + 1
        return QuantumGate(z_matrix, targets=[target], controls=controls, name=f'{n_qubits}-control mcz')