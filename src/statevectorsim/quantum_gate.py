import numpy as np
import math
from typing import Union, List, Dict
from .quantum_state import QuantumState
from scipy.sparse import csr_matrix, identity, block_diag, kron, isspmatrix_csr

class QuantumGate:
    def __init__(self, matrix: np.ndarray, targets: list[int], name: str = 'CustomGate'):
        # robust check
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

    def __str__(self):
        """String representation of the QuantumGate."""
        n_qubits = len(self.targets)
        targets_str = ", ".join(map(str, self.targets))
        return f"{n_qubits}-Qubit {self.name} Gate on Qubits [{targets_str}]"

    # -------------------------------------
    #           Helper Methods
    # -------------------------------------

    @staticmethod
    def _create_single_gates(targets: Union[int, List[int]], matrix: np.ndarray, name: str = 'CustomSingleGate') -> List['QuantumGate']:
        """Helper to apply single-qubit gate to one or multiple qubits."""
        if isinstance(targets, int):
            targets = [targets]

        # returns a list of QuantumGate objects. If only one target, it's a list with one item.
        return [QuantumGate(matrix, [t], name) for t in targets]

    @staticmethod
    def _create_controlled_gate(control: int, target: int, matrix: np.ndarray, name: str = 'CustomMultiGate') -> 'QuantumGate':
        """Helper function to create a 2-qubit gate instance."""
        return QuantumGate(matrix, [control, target], name)

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

        # 1. Define Quibit Order and Permutation
        active_qubits = set(self.targets) | set(self.controls)
        non_active_qubits = sorted([i for i in range(N) if i not in active_qubits])

        # New order for efficient construction: [Non-Active, Controls, Targets]
        new_order = non_active_qubits + sorted(self.controls) + sorted(self.targets)
        original_order = list(range(N))

        # Get the Permutation Matrix P (converts original basis to new basis)
        P = _generate_permutation_matrix(N, original_order, new_order)
        P_T = P.transpose()  # Transpose is the inverse for P

        # 2. Build the Active Subspace Gate (G_active)

        num_active = len(active_qubits)
        dim_active = 2 ** num_active

        if not self.controls:
            # Non-Controlled Gate: G_active is just the gate matrix U
            G_active = csr_matrix(self.matrix)
        else:
            # Controlled Gate: G_active = I on control=0 blocks, U on control=1 block

            # The matrix U is 2^k x 2^k
            U_matrix_csr = csr_matrix(self.matrix)
            dim_k = 2 ** len(self.targets)

            # The control part of the subspace (2^c x 2^c)
            dim_control = 2 ** len(self.controls)

            # G_active is a block matrix of size 2^(c+k) x 2^(c+k)
            # It's an Identity matrix everywhere except the block where ALL controls are 1

            # Create a 2^(c+k) x 2^(c+k) sparse identity matrix
            G_active = identity(dim_active, dtype=complex, format='lil')

            # Identify the block offset: it's the index where all controls are |1>
            # The last block is at index 2^c - 1, starting at row/col (2^c - 1) * 2^k
            block_start_index = (dim_control - 1) * dim_k

            # Overwrite the control=|1> block with the gate matrix U
            # The block is from [block_start_index : block_start_index + dim_k]
            G_active[block_start_index: block_start_index + dim_k,
            block_start_index: block_start_index + dim_k] = U_matrix_csr

            G_active = G_active.tocsr()

        # 3. Embed G_active into the Full Permuted Space (U')
        # U' = I_non_active \otimes G_active
        I_non_active = identity(2 ** len(non_active_qubits), dtype=complex, format='csr')
        U_prime = kron(I_non_active, G_active, format='csr')

        # 4. Revert Permutation: U = P^T * U' * P
        # This gives the final, full, unpermuted sparse matrix.
        U_full = P_T.dot(U_prime.dot(P))

        return U_full.tocsr()

    # -------------------------------------------
    #    Apply Gate Logic (Speed-up bottleneck)
    # -------------------------------------------

    def _apply_tensor(self, state: np.ndarray, n: int, k: int) -> np.ndarray:
        """
        Multi-qubit gate application using NumPy's efficient tensor reshape and permutation (k >= 1).
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
        Multi-qubit gate application using generalized index-based iteration (k >= 1).
        """

        gate_matrix = self.matrix
        dim_k = 2 ** k
        new_state = state.copy()

        # Sort targets to ensure the statevector block's internal order matches the gate matrix's order.
        sorted_targets = sorted(self.targets)

        # Pre-calculate the index offsets
        index_offsets = np.zeros(dim_k, dtype=int)
        for m in range(dim_k):
            offset = 0
            for j, t in enumerate(sorted_targets):
                # If the j-th bit of m is 1, add 1 << t (the state index bit) to offset
                if (m >> j) & 1:
                    offset |= (1 << t)
            index_offsets[m] = offset

        # iteration block mask. sum of all target masks. determines indices 'i' where all targets are |0>.
        control_mask = sum(1 << t for t in self.targets)

        # Iterate over the 2^(n-k) blocks
        size = 1 << n

        for i in range(size):
            # start iteration at the index 'i' where all target qubits are |0>
            if i & control_mask:
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
        Apply k-qubit QuantumGate using the high-performance, C based, SciPy CSR matrix
        multiplication approach.

        If the gate's definition requires a full 2^N x 2^N gate matrix,
        this approach is efficient because the single dot product is C-accelerated.
        """

        # Build full sparse matrix U_full (2^N x 2^N) for gate.
        U_full_sparse = self._build_sparse_full_matrix(quantum_state.n)

        # Apply gate using the optimised SciPy dot product.
        new_sparse_state = U_full_sparse.dot(quantum_state.sparse_state.transpose()).transpose().tocsr()

        # Update sparse rep and clean
        quantum_state.sparse_state = new_sparse_state
        quantum_state.clean_sparse()


    def apply(self, quantum_state, method: str = 'tensor'):
        """
        Apply k-qubit QuantumGate to an n-qubit QuantumState using the specified method.

        Args:
            quantum_state: The QuantumState object.
            method (str): implementation strategy.
                            - 'tensor' (NumPy reshape/transpose)
                            - 'bitmask' (optimized index-based iteration).
        """

        n = quantum_state.n
        k = len(self.targets)

        # Check if targets are valid (within [0, n-1])
        if not all(0 <= t < n for t in self.targets):
            raise ValueError(
                f"Target qubits {self.targets} are out of bounds "
                f"for an {n}-qubit state."
            )

        original_mode = quantum_state.mode

        if original_mode == 'sparse':
            if method in ['tensor', 'bitmask']:
                # The user specified a dense method, so convert to dense first
                quantum_state.to_dense()
                state = quantum_state.state.copy()
            else:
                # Use the new sparse implementation if the state is sparse
                self._apply_sparse(quantum_state)
                # Ensure the mode is explicitly set to sparse
                quantum_state.mode = 'sparse'
                return  # Exit early as the state is updated

        else:  # original_mode == 'dense'
            state = quantum_state.state.copy()

        # --- Dense Application ---
        if method == 'tensor':
            new_state = self._apply_tensor(state, n, k)
        elif method == 'bitmask':
            new_state = self._apply_bitmask(state, n, k)
        elif method == 'sparse':
            raise ValueError(
                "Cannot use 'sparse' method on a dense QuantumState. "
                "Ensure the QuantumState's mode is set to 'sparse' or use to_sparse() first."
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'tensor', 'bitmask', or 'sparse'.")

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
        """CNOT gate (2 qubits) - Control on first target, Target on second."""
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=complex)
        return QuantumGate._create_controlled_gate(control, target, matrix, 'cx')

    @staticmethod
    def cy(control: int, target: int):
        """Controlled-y gate (2 qubits) - Applies y if both qubits are |1>"""
        matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return QuantumGate._create_controlled_gate(control, target, matrix, 'cy')

    @staticmethod
    def cz(control: int, target: int):
        """Controlled-Z gate (2 qubits) - Applies Z if both qubits are |1>"""
        # CZ matrix is diagonal with [1, 1, 1, -1]
        matrix = np.diag([1, 1, 1, -1]).astype(complex)
        return QuantumGate._create_controlled_gate(control, target, matrix, 'cz')

    @staticmethod
    def swap(q1: int, q2: int):
        """SWAP gate (2 qubits). Basis: |00>, |01>, |10>, |11>"""
        matrix = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]], dtype=complex)
        return QuantumGate._create_controlled_gate(q1, q2, matrix, 'swap')  # Using the 2-qubit helper here

    @staticmethod
    def cu(control: int, target_qubits: list[int], unitary_matrix: np.ndarray, k: int = 1):
        """ Creates a Controlled-U^k gate. """

        # num target qubits
        m_qubits = len(target_qubits)

        if unitary_matrix.shape != (2 ** m_qubits, 2 ** m_qubits):
            raise ValueError(
                f"Unitary matrix size must be 2^m x 2^m where m is the number of target qubits ({m_qubits})."
            )

        # Compute U^k
        U_k = np.linalg.matrix_power(unitary_matrix, k)

        # Determine the full set of qubits involved
        all_qubits = sorted(list(set([control] + target_qubits)))
        n_total = len(all_qubits)

        # Identity matrix for the target register
        I_target = QuantumGate._id(m_qubits)

        # Construct the Controlled-U matrix (C-U) in the computational basis:
        # projection operators:
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)

        # tensor product P0 I_target
        P0_I = np.kron(P0, I_target)

        # tensor product P1 U^k
        P1_U = np.kron(P1, U_k)

        # Full C-U matrix (before permutation/re-indexing)
        CU_matrix_unpermuted = P0_I + P1_U

        # The indices for the QuantumGate object must be all qubits involved.
        gate_qubits = [control] + target_qubits

        return QuantumGate(CU_matrix_unpermuted, gate_qubits)

    # ----------------------------------------------
    #    Controlled Rotation Gates (CRX, CRY, CRZ)
    # ----------------------------------------------

    @staticmethod
    def crx(control: int, target: int, theta: float) -> 'QuantumGate':
        """
        Controlled-Rx gate.
        Applies Rx(theta) to the target qubit if the control qubit is |1>.

        The 4x4 matrix assumes targets are [control, target] (basis |00>, |01>, |10>, |11>).
        """
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)

        # Identity block for control=0 (top-left 2x2)
        # R_x(theta) block for control=1 (bottom-right 2x2)
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -1j * s],
            [0, 0, -1j * s, c]
        ], dtype=complex)

        return QuantumGate._create_controlled_gate(control, target, matrix, f'crx({theta})')

    @staticmethod
    def cry(control: int, target: int, theta: float) -> 'QuantumGate':
        """
        Controlled-Ry gate.
        Applies Ry(theta) to the target qubit if the control qubit is |1>.

        The 4x4 matrix assumes targets are [control, target] (basis |00>, |01>, |10>, |11>).
        """
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)

        # Identity block for control=0
        # R_y(theta) block for control=1
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c]
        ], dtype=complex)

        return QuantumGate._create_controlled_gate(control, target, matrix, f'cry({theta})')

    @staticmethod
    def crz(control: int, target: int, theta: float) -> 'QuantumGate':
        """Controlled-Rz gate Rz(theta)."""
        half_theta = theta / 2.0
        e_neg = np.exp(-1j * half_theta)
        e_pos = np.exp(1j * half_theta)
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, e_neg, 0],
            [0, 0, 0, e_pos]
        ], dtype=complex)
        return QuantumGate._create_controlled_gate(control, target, matrix, f'crz({theta})')

    @staticmethod
    def crp(control: int, target: int, theta: float):
        """ Controlled Phase Gate (CRP or CP(theta)).
        Applies a phase shift of exp(i*theta) to the |11> component.
        """

        # control and target are already type-hinted as int
        if control == target:
            raise ValueError("Control and target qubits must be distinct.")

        # The matrix is defined for sorted indices |00>, |01>, |10>, |11>
        matrix = np.eye(4, dtype=complex)
        matrix[3, 3] = np.exp(1j * theta)

        return QuantumGate._create_controlled_gate(control, target, matrix, f'crp({theta})')


    # -----------------------------------------
    #   Multi-Controlled Gates (MCX, MCY, MCZ)
    # -----------------------------------------

    @staticmethod
    def mcx(controls: List[int], target: int):
        """ Multi-Controlled X (MCX) or N-Controlled NOT gate. """

        all_targets = controls + [target]
        n_qubits = len(all_targets)

        if n_qubits < 2:
            raise ValueError("MCX requires at least one control and one target qubit.")

        dim = 2 ** n_qubits
        matrix = np.eye(dim, dtype=complex)

        # The bottom-right 2x2 block corresponds to all controls = 1.
        index_0 = dim - 2  # index of |11...10>
        index_1 = dim - 1  # index of |11...11>

        # apply the Pauli-X matrix to the block, effectively swapping the two states
        matrix[index_0, index_0] = 0
        matrix[index_0, index_1] = 1
        matrix[index_1, index_0] = 1
        matrix[index_1, index_1] = 0

        return QuantumGate(matrix, all_targets, f'{n_qubits}-control mcx')

    @staticmethod
    def mcy(controls: List[int], target: int):
        """ Multi-Controlled Y (MCY) gate. """
        all_targets = controls + [target]
        n_qubits = len(all_targets)

        if n_qubits < 2:
            raise ValueError("MCY requires at least one control and one target qubit.")

        dim = 2 ** n_qubits
        matrix = np.eye(dim, dtype=complex)

        # The index block corresponding to all controls = 1.
        index_0 = dim - 2  # index of |11...10>
        index_1 = dim - 1  # index of |11...11>

        # apply Pauli-Y matrix
        matrix[index_0, index_0] = 0
        matrix[index_0, index_1] = -1j
        matrix[index_1, index_0] = 1j
        matrix[index_1, index_1] = 0

        return QuantumGate(matrix, all_targets, f'{n_qubits}-control mcy')

    @staticmethod
    def mcz(controls: List[int], target: int):
        """ Multi-Controlled Z (MCZ) gate. """
        all_targets = controls + [target]
        n_qubits = len(all_targets)

        if n_qubits < 2:
            raise ValueError("MCZ requires at least one control and one target qubit.")

        dim = 2 ** n_qubits
        matrix = np.eye(dim, dtype=complex)

        # index corresponding to all qubits being |1> is the last index
        index_all_ones = dim - 1  # index of |11..11>

        # apply a phase flip (-1) only when all inputs are 1
        matrix[index_all_ones, index_all_ones] = -1

        return QuantumGate(matrix, all_targets, f'{n_qubits}-control mcz')
