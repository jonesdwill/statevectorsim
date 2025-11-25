import numpy as np
import math
from typing import Union, List
from .quantum_state import QuantumState

class QuantumGate:
    def __init__(self, matrix: np.ndarray, targets: list[int]):
        # robust check
        expected_dim = 2 ** len(targets)
        if matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Matrix shape {matrix.shape} does not match "
                f"expected size ({expected_dim}, {expected_dim}) "
                f"for {len(targets)} targets."
            )
        # gate assignment
        self.matrix = matrix
        self.targets = targets

    def apply(self, quantum_state):
        """
        Apply k-qubit QuantumGate to an n-qubit QuantumState.
        - k=1: bitmask iteration.
        - k>=2: tensor reshape and permutation.
        """
        # get statevector, n and k.
        n = quantum_state.n
        state = quantum_state.state.copy()
        k = len(self.targets)

        # ----------------------------------------------------
        #               Single-qubit gate (k=1)
        # ----------------------------------------------------
        if k == 1:
            t = self.targets[0] # get single target
            target_mask = 1 << t # bitmask of 2^t
            size = 1 << n # bitmask of 2^n
            gate_matrix = self.matrix # 2x2 matrix for single qubit gate

            # loop over qubits
            for i in range(size):
                # only iterate over 2-dim subspace once (by picking where target qubit is |0>)
                if i & target_mask:
                    continue

                # |0> and |1> indices affected qubit
                j0 = i
                j1 = i | target_mask

                # get qubit amplitudes
                block = np.array([state[j0], state[j1]])

                # apply 2x2 gate to amplitudes
                new_block = gate_matrix @ block

                # write back
                state[j0] = new_block[0]
                state[j1] = new_block[1]

            # update state
            quantum_state.state = state
            return

        # -------------------------------------------------------
        # Multi-qubit gates (k >= 2) - Generalized Tensor Method
        # -------------------------------------------------------
        elif k >= 2:

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
            tensor_state = self.matrix @ tensor_state

            # reshape back to original tensor shape
            tensor_state = tensor_state.reshape([2] * k + list(dims_other))

            # inverse qubit permutation
            axes_original = np.argsort(axes_permuted)
            tensor_state = np.transpose(tensor_state, axes_original)

            # flatten and update the quantum state
            quantum_state.state = tensor_state.reshape(-1)
            return

        else:
            raise ValueError("Gate targets list is empty.")



    # -------------------------------------
    #           Helper Methods
    # -------------------------------------

    @staticmethod
    def _create_single_gates(targets: Union[int, List[int]], matrix: np.ndarray) -> List['QuantumGate']:
        """Helper to apply single-qubit gate to one or multiple qubits."""
        if isinstance(targets, int):
            targets = [targets]

        # returns a list of QuantumGate objects. If only one target, it's a list with one item.
        return [QuantumGate(matrix, [t]) for t in targets]

    @staticmethod
    def _create_controlled_gate(control: int, target: int, matrix: np.ndarray) -> 'QuantumGate':
        """Helper function to create a 2-qubit gate instance."""
        return QuantumGate(matrix, [control, target])



    # -------------------------------------
    #           Standard Gates
    # -------------------------------------

    @staticmethod
    def x(targets: Union[int, List[int]]):
        """Single or multiple Pauli-X gates."""
        matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix)

    @staticmethod
    def y(targets: Union[int, List[int]]):
        """Single or multiple Pauli-Y gates."""
        matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix)

    @staticmethod
    def z(targets: Union[int, List[int]]):
        """Single or multiple Pauli-Z gates."""
        matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix)

    @staticmethod
    def h(targets: Union[int, List[int]]):
        """Single or multiple Hadamard gates."""
        matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix)

    @staticmethod
    def i(targets: Union[int, List[int]]):
        """Identity gate"""
        matrix = np.eye(2, dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix)



    # -------------------------------------
    #           Rotation Gates
    # -------------------------------------

    @staticmethod
    def rx(targets: Union[int, List[int]], theta: float):
        """Single or multiple Pauli-X rotation gates. Rx(theta)"""
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)
        matrix = np.array([[c, -1j * s],
                           [-1j * s, c]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix)

    @staticmethod
    def ry(targets: Union[int, List[int]], theta: float):
        """Single or multiple Pauli-Y rotation gates. Ry(theta)"""
        half_theta = theta / 2.0
        c = math.cos(half_theta)
        s = math.sin(half_theta)
        matrix = np.array([[c, -s],
                           [s, c]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix)

    @staticmethod
    def rz(targets: Union[int, List[int]], theta: float):
        """Single or multiple Pauli-Z rotation gates. Rz(theta)"""
        half_theta = theta / 2.0
        matrix = np.array([[np.exp(-1j * half_theta), 0],
                           [0, np.exp(1j * half_theta)]], dtype=complex)
        return QuantumGate._create_single_gates(targets, matrix)



    # -------------------------------------
    #           Controlled Gates
    # -------------------------------------

    @staticmethod
    def cx(control: int, target: int):
        """CNOT gate (2 qubits) - Control on first target, Target on second."""
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=complex)
        return QuantumGate._create_controlled_gate(control, target, matrix)

    @staticmethod
    def cz(control: int, target: int):
        """Controlled-Z gate (2 qubits) - Applies Z if both qubits are |1>"""
        # CZ matrix is diagonal with [1, 1, 1, -1]
        matrix = np.diag([1, 1, 1, -1]).astype(complex)
        return QuantumGate._create_controlled_gate(control, target, matrix)

    @staticmethod
    def swap(q1: int, q2: int):
        """SWAP gate (2 qubits). Basis: |00>, |01>, |10>, |11>"""
        matrix = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]], dtype=complex)
        return QuantumGate._create_controlled_gate(q1, q2, matrix)  # Using the 2-qubit helper here



    # -------------------------------------
    #      Controlled Rotation Gates
    # -------------------------------------

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

        return QuantumGate._create_controlled_gate(control, target, matrix)

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

        return QuantumGate._create_controlled_gate(control, target, matrix)


    @staticmethod
    def crz(control: int, target: int, theta: float) -> 'QuantumGate':
        """Controlled-Rz gate Rz(theta)."""
        half_theta = theta / 2.0
        # Use np.exp for complex exponentiation
        e_neg = np.exp(-1j * half_theta)
        e_pos = np.exp(1j * half_theta)
        matrix = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0],
            [0, 0, e_neg, 0], [0, 0, 0, e_pos]
        ], dtype=complex)
        return QuantumGate._create_controlled_gate(control, target, matrix)



    # -------------------------------------
    #   Multi-Controlled (N-qubit) Gates
    # -------------------------------------

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

        return QuantumGate(matrix, all_targets)

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

        return QuantumGate(matrix, all_targets)

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

        return QuantumGate(matrix, all_targets)
