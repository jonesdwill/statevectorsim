import numpy as np

def embed_single_qubit_gate(n: int, gate: np.ndarray, targets: list[int]) -> np.ndarray:
    """
    Embed a single-qubit gate into an n-qubit system at arbitrary targets.

    Args:
        n: total number of qubits
        gate: 2x2 numpy array representing the single-qubit gate
        targets: list of qubit indices to apply the gate to (0=MSB)

    Returns:
        2^n x 2^n numpy array representing the combined gate
    """

    dim = 2 ** n
    full_matrix = np.eye(dim, dtype=complex)

    for t in targets:
        target_bit = 1 << t
        new_matrix = np.zeros_like(full_matrix)

        # Iterate over all basis indices
        for i in range(dim):
            j = i ^ target_bit  # flip the target qubit
            new_matrix[i, i] += gate[0, 0]
            new_matrix[i, j] += gate[0, 1] if (i & target_bit) == 0 else gate[1, 1]
        full_matrix = new_matrix @ full_matrix

    return full_matrix