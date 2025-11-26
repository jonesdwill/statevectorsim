import math
import numpy as np
from statevectorsim import QuantumState, QuantumCircuit, QuantumGate
from statevectorsim.utils import plot_bloch_spheres, format_statevector, circuit_to_ascii
from numpy import testing

# ==============================================================================
#                      Testing Utility Functions
# ==============================================================================

def run_test_and_plot(title: str, circuit: QuantumCircuit, target_qubits: list[int] = None):
    """
    Initializes a QuantumState (all |0>), runs the circuit, and visualizes
    the result on the Bloch spheres for all qubits in the circuit.
    """
    n_qubits = circuit.n
    print("=" * 70)
    print(f"RUNNING TEST: {title} ({n_qubits} Qubits)")

    # Initialize the state to |0...0>
    state = QuantumState(n_qubits)

    # Run the circuit
    print("Applying Gates...")
    circuit.run(state)

    # Print the resulting statevector for verification
    print(f"Final State Vector: {state.state}")

    # Visualize the Resulting State
    print(f"Plotting Bloch Spheres for {n_qubits} Qubits...")

    # The plot_bloch_spheres function is expected to handle the state vector
    plot_bloch_spheres(state.state)

    print("-" * 70)


# ==============================================================================
#                      Single-Qubit Gate Tests (1 Qubit)
# ==============================================================================

def test_i_gate():
    """Test Identity (I) gate: |0> -> |0> (no change)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.i(0))
    run_test_and_plot("1. I Gate: |0> -> |0>", qc, target_qubits=[0])

def test_x_gate():
    """Test Pauli-X (NOT) gate: |0> -> |1> (Flips the state)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.x(0))
    run_test_and_plot("2. X Gate: |0> -> |1> (+Z to -Z)", qc, target_qubits=[0])

def test_y_gate():
    """Test Pauli-Y gate: |0> -> i|1> (Rotates by pi about Y-axis to -Z)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.y(0))
    run_test_and_plot("3. Y Gate: |0> -> i|1> (+Z to -Z, with global phase)", qc, target_qubits=[0])

def test_z_gate():
    """Test Pauli-Z gate: |0> -> |0> (No change on |0>, 180 deg phase flip on |1>)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.z(0))
    run_test_and_plot("4. Z Gate: |0> -> |0>", qc, target_qubits=[0])

def test_h_gate():
    """Test Hadamard (H) gate: |0> -> |+> (Superposition, +X axis)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.h(0))
    run_test_and_plot("5. H Gate: |0> -> |+> (+X axis)", qc, target_qubits=[0])

def test_s_gate():
    """Test Phase (S) gate on |+> state: S|+> = (|0> + i|1>)/sqrt(2) (+Y axis)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.h(0)) # |0> -> |+>
    # The S gate (Z rotation by pi/2) rotates |+> from +X to +Y
    qc.add_gate(QuantumGate.s(0))
    run_test_and_plot("6. S Gate on |+> state: +X -> +Y", qc, target_qubits=[0])

def test_t_gate():
    """Test T gate on |+> state: Rotates phase by pi/4 (towards +Y)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.h(0)) # |0> -> |+>
    qc.add_gate(QuantumGate.t(0))
    run_test_and_plot("7. T Gate on |+> state: Phase rotation by pi/4", qc, target_qubits=[0])

def test_rx_pi_2():
    """Test Rx(pi/2) gate: Rotates |0> 90 deg about X-axis to -Y axis."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.rx(0, math.pi / 2))
    run_test_and_plot("8. Rx(pi/2): |0> -> -|y>", qc, target_qubits=[0])

def test_ry_pi_2():
    """Test Ry(pi/2) gate: Rotates |0> 90 deg about Y-axis to +X axis (|0> -> |+>)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.ry(0, math.pi / 2))
    run_test_and_plot("9. Ry(pi/2): |0> -> |+> (+X)", qc, target_qubits=[0])

def test_rz_pi():
    """Test Rz(pi) gate on |+>: Rotates |+> 180 deg about Z-axis to |-> (-X axis)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.h(0)) # |0> -> |+>
    qc.add_gate(QuantumGate.rz(0, math.pi))
    run_test_and_plot("10. Rz(pi) on |+>: +X -> -X", qc, target_qubits=[0])


# ==============================================================================
#                      Two-Qubit Controlled/Swap Gates
# ==============================================================================

def test_bell_state_cx():
    """Test CNOT (CX) gate by creating the Bell state |Φ+>."""
    qc = QuantumCircuit(2)
    qc.add_gate(QuantumGate.h(0))    # |00> -> (|00> + |10>)/sqrt(2)
    qc.add_gate(QuantumGate.cx(0, 1)) # -> (|00> + |11>)/sqrt(2) (Bell state)

    print(circuit_to_ascii(qc))

    # Entangled state should have both qubits in a mixed state (vector at center)
    run_test_and_plot("11. Bell State (|Φ+>) using H & CX (Q0=C, Q1=T)", qc, target_qubits=[0, 1])

def test_cz_gate():
    """Test CZ gate by acting on the Bell state |Φ+> to get |Ψ->."""
    qc = QuantumCircuit(2)
    qc.add_gate(QuantumGate.h(0))    # H(0)
    qc.add_gate(QuantumGate.cx(0, 1)) # CX(0, 1) -> |Φ+>
    qc.add_gate(QuantumGate.cz(0, 1)) # CZ(|Φ+>) = |Ψ-> = (|00>-|11>)/sqrt(2)

    print(circuit_to_ascii(qc))

    run_test_and_plot("12. CZ Gate on Bell State (Still entangled)", qc, target_qubits=[0, 1])

def test_swap_gate():
    """Test SWAP gate: Swap states of |01> to |10>."""
    qc = QuantumCircuit(2)
    # Prepare Qubit 1 in |1> state: |00> -> |01>
    qc.add_gate(QuantumGate.x(1))
    # Apply SWAP(0, 1) to get |10>
    qc.add_gate(QuantumGate.swap(0, 1))

    print(circuit_to_ascii(qc))

    # Qubit 0 should now be |1> (down) and Qubit 1 should be |0> (up)
    run_test_and_plot("13. SWAP Gate: |01> -> |10>", qc, target_qubits=[0, 1])


# ==============================================================================
#                      Controlled Rotation Gates (2 Qubits)
# ==============================================================================

def test_crx_gate():
    """Test CRX(pi) gate: Apply X (180 deg rotation) to target if control is |1>."""
    qc = QuantumCircuit(2)
    qc.add_gate(QuantumGate.x(0))      # Set control Q0 to |1>. State: |10>
    qc.add_gate(QuantumGate.crx(0, 1, math.pi)) # Should apply X to Q1: |10> -> |11>

    # Both Q0 and Q1 should be pointing down (-Z axis)
    run_test_and_plot("14. CRX(pi) on |10>: |10> -> |11>", qc, target_qubits=[0, 1])

def test_cry_gate():
    """Test CRY(pi) gate: Apply Y (180 deg rotation) to target if control is |1>."""
    qc = QuantumCircuit(2)
    qc.add_gate(QuantumGate.x(0))      # Set control Q0 to |1>. State: |10>
    qc.add_gate(QuantumGate.cry(0, 1, math.pi)) # Should apply Y to Q1: |10> -> i|11>

    # Both Q0 and Q1 should be pointing down (-Z axis)
    run_test_and_plot("15. CRY(pi) on |10>: |10> -> i|11>", qc, target_qubits=[0, 1])

def test_crz_gate():
    """Test CRZ(pi) gate: Apply Z (180 deg rotation) to target if control is |1>."""
    qc = QuantumCircuit(2)
    # Prepare a state where RZ is noticeable on Q1: |1+> = (|10> + |11>)/sqrt(2)
    qc.add_gate(QuantumGate.x(0))
    qc.add_gate(QuantumGate.h(1))      # State: |1+>

    # CRZ(pi) should apply Z to Q1: |1+> -> |1-> (Q1 moves from +X to -X)
    qc.add_gate(QuantumGate.crz(0, 1, math.pi))

    # Q0: |1> (-Z); Q1: |-> (-X)
    run_test_and_plot("16. CRZ(pi) on |1+>: Q1 is flipped +X -> -X", qc, target_qubits=[0, 1])


# ==============================================================================
#                      Multi-Controlled Gates (3 Qubits)
# ==============================================================================

def test_mcx_gate():
    """Test Toffoli (CCX) gate: Multi-Controlled X (2 controls, 1 target)."""
    qc = QuantumCircuit(3)
    # Set controls Q0 and Q1 to |1>. State: |110>
    qc.add_gate(QuantumGate.x([0, 1]))
    # Apply MCX(0, 1, target=2): |110> -> |111>
    qc.add_gate(QuantumGate.mcx([0, 1], 2))

    # All qubits should be pointing down (-Z axis)
    run_test_and_plot("17. MCX (Toffoli): |110> -> |111>", qc, target_qubits=[0, 1, 2])

def test_mcy_gate():
    """Test Multi-Controlled Y gate (2 controls, 1 target)."""
    qc = QuantumCircuit(3)
    # Set controls Q0 and Q1 to |1>. State: |110>
    qc.add_gate(QuantumGate.x([0, 1]))
    # Apply MCY(0, 1, target=2): |110> -> i|111>
    qc.add_gate(QuantumGate.mcy([0, 1], 2))

    # All qubits should be pointing down (-Z axis)
    run_test_and_plot("18. MCY: |110> -> i|111>", qc, target_qubits=[0, 1, 2])

def test_mcz_gate():
    """Test Multi-Controlled Z gate (2 controls, 1 target)."""
    qc = QuantumCircuit(3)
    # Prepare state |11+> where RZ is noticeable on Q2:
    qc.add_gate(QuantumGate.x([0, 1]))
    qc.add_gate(QuantumGate.h(2))      # State: |11+>

    # MCZ should apply Z to Q2: |11+> -> -|11-> (Q2 moves from +X to -X)
    qc.add_gate(QuantumGate.mcz([0, 1], 2))

    # Q0, Q1: |1> (-Z); Q2: |-> (-X)
    run_test_and_plot("19. MCZ on |11+>: Q2 is flipped +X -> -X", qc, target_qubits=[0, 1, 2])


# ==============================================================================
#                                QFT TESTING
# ==============================================================================

def test_qft_decomposition(n_qubits: int, initial_index: int):
    """
    Test the QFT implementation by applying it to a basis state |x> and checking against analytical result.
    """

    # Pad the index for printing
    x_str = bin(initial_index)[2:].zfill(n_qubits)
    print(f"--- Testing QFT Decomposition for |{x_str}> ({n_qubits} qubits) ---")

    # Define the input state |x>
    initial_state = QuantumState(n_qubits)
    initial_state.basis_state(initial_index)

    # Build the QFT circuit. QFT swaps endian so reverse.
    qft_circuit = QuantumCircuit.qft(n_qubits, swap_endian=True)

    # Run the circuit
    final_state = qft_circuit.run(initial_state)

    print(circuit_to_ascii(qft_circuit))

    # Define the expected output state (analytical result)
    expected_state = np.zeros(2**n_qubits, dtype=complex)
    N = 2**n_qubits
    x = initial_index

    # Calculate expected amplitudes for each basis state |k>
    for k in range(N):
        # Calculate the phase: 2 * pi * x * k / N
        phase_angle = 2 * np.pi * x * k / N
        # Amplitude is (1/sqrt(N)) * exp(i * phase_angle)
        expected_state[k] = (1 / np.sqrt(N)) * np.exp(1j * phase_angle)

    # Assert: Check if the final state matches the expected state
    tolerance = 1e-7
    assert np.allclose(final_state.state, expected_state, atol=tolerance), (
        f"QFT state mismatch for {n_qubits} qubits on |{x_str}>.\n"
        f"Expected:\n{expected_state}\n"
        f"Got:\n{final_state.state}\n"
    )

    print(f"QFT Test ({n_qubits} qubits on |{x_str}>) PASSED.")
    print("--- QFT Test Results (First 8 amplitudes) ---")
    print(f"Final State Vector: {final_state.state[:8]}...")

    plot_bloch_spheres(final_state.state)


# ==============================================================================
#                      QPE Testing Utility
# ==============================================================================

def get_most_likely_phase(state: QuantumState, t_qubits: int, target_val: int) -> float:
    """ Analyzes the final statevector from a QPE circuit to determine the most likely phase. """

    probabilities = state.get_probabilities()
    n_qubits = state.n
    m_qubits = n_qubits - t_qubits
    target_offset = target_val * (2**t_qubits)

    # Isolate the probabilities corresponding to the correct target register outcome
    counting_probabilities = probabilities[target_offset : target_offset + (2**t_qubits)]

    if np.sum(counting_probabilities) < 0.9:
        print("Warning: Target register outcome was not well resolved")

    # Find the index k with max probability
    max_index_k = np.argmax(counting_probabilities)

    # Estimate phase from max probability
    estimated_phase = max_index_k / (2**t_qubits)

    return estimated_phase


def test_qpe_s_gate():
    """ Tests Quantum Phase Estimation for the S gate (phase phi = 1/4) """

    print("\n" + "=" * 70)
    print("RUNNING TEST: QUANTUM PHASE ESTIMATION (S Gate)")

    t_qubits = 4  # Precision 1/16
    m_qubits = 1

    # U|1> = i|1> = e^(i * 2 * pi * 1/4) -> Expected phase: 0.25
    S_matrix = np.array([[1, 0], [0, 1j]], dtype=complex)

    target_qubit_index = t_qubits
    target_prep_gates = [QuantumGate.x(target_qubit_index)]
    target_val = 1  # |1> state

    qpe_circuit = QuantumCircuit.qpe(
        t_qubits=t_qubits, unitary_matrix=S_matrix, m_qubits=m_qubits, target_initial_state_gates=target_prep_gates
    )

    initial_state = QuantumState(qpe_circuit.n)
    print(f"Applying {len(qpe_circuit.gates)} Gates to {qpe_circuit.n}-qubit state...")
    qpe_circuit.run(initial_state)

    estimated_phi = get_most_likely_phase(initial_state, t_qubits, target_val)
    expected_phi = 0.25
    is_correct = np.isclose(estimated_phi, expected_phi)

    print(circuit_to_ascii(qpe_circuit))

    print(f"Expected Phase: {expected_phi:.4f}")
    print(f"Estimated Phase (k / 2^t): {estimated_phi:.4f}")
    print(f"QPE Test PASSED: {is_correct}")
    print("-" * 70)
    print(f"Final State: {format_statevector(initial_state.state)}")


def test_qpe_t_gate():
    """ Tests Quantum Phase Estimation for the T gate (phase phi = 1/8) """

    print("\n" + "=" * 70)
    print("RUNNING TEST: QUANTUM PHASE ESTIMATION (T Gate)")

    t_qubits = 4  # Precision 1/16
    m_qubits = 1

    T_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    target_qubit_index = t_qubits
    target_prep_gates = [QuantumGate.x(target_qubit_index)]
    target_val = 1  # |1> state

    qpe_circuit = QuantumCircuit.qpe(
        t_qubits=t_qubits, unitary_matrix=T_matrix, m_qubits=m_qubits, target_initial_state_gates=target_prep_gates
    )

    initial_state = QuantumState(qpe_circuit.n)
    print(f"Applying {len(qpe_circuit.gates)} Gates to {qpe_circuit.n}-qubit state...")
    initial_state.state[2 ** t_qubits] = 1.0  # Ensures target is |1> at start
    initial_state.state[0] = 0.0  # set |0> to 0 for initial state |1>
    qpe_circuit.run(initial_state)

    estimated_phi = get_most_likely_phase(initial_state, t_qubits, target_val)
    expected_phi = 0.125

    # expect k=2 for t=4 (2/16 = 0.125)
    is_correct = np.isclose(estimated_phi, expected_phi)

    print(circuit_to_ascii(qpe_circuit))

    print(f"Expected Phase: {expected_phi:.4f}")
    print(f"Estimated Phase (k / 2^t): {estimated_phi:.4f}")
    print(f"QPE Test PASSED: {is_correct}")
    print("-" * 70)
    print(f"Final State: {format_statevector(initial_state.state)}")


def test_qpe_approx_pi_3():
    """ Tests QPE for a phase (1/3) that is NOT a clean power of two fraction. """

    print("\n" + "=" * 70)
    print("RUNNING TEST: QUANTUM PHASE ESTIMATION (Approx. Phase 1/3)")

    t_qubits = 6  # Higher precision 1/64
    m_qubits = 1

    # Phase U|1> = e^(i * 2 * pi * 1/3)|1> -> Expected phase: 1/3 ≈ 0.33333
    expected_phi = 1 / 3
    U_matrix = np.array([[1, 0], [0, np.exp(1j * 2 * np.pi / 3)]], dtype=complex)

    target_qubit_index = t_qubits
    target_prep_gates = [QuantumGate.x(target_qubit_index)]
    target_val = 1  # |1> state

    qpe_circuit = QuantumCircuit.qpe(
        t_qubits=t_qubits, unitary_matrix=U_matrix, m_qubits=m_qubits, target_initial_state_gates=target_prep_gates
    )

    initial_state = QuantumState(qpe_circuit.n)
    print(f"Applying {len(qpe_circuit.gates)} Gates to {qpe_circuit.n}-qubit state...")
    qpe_circuit.run(initial_state)

    estimated_phi = get_most_likely_phase(initial_state, t_qubits, target_val)

    # For 1/3, the closest fraction is 21/64 ≈ 0.328125 or 22/64 ≈ 0.34375
    # The true value is 0.33333. test for close approximation.
    is_correct = np.isclose(estimated_phi, expected_phi, atol=0.01)  # Check within 1% error

    print(circuit_to_ascii(qpe_circuit))

    print(f"True Phase: {expected_phi:.5f}")
    print(f"Estimated Phase (k / 2^t): {estimated_phi:.5f}")
    print(f"QPE Test PASSED: {is_correct} (Closest fraction is {int(estimated_phi * (2 ** t_qubits))}/{2 ** t_qubits})")
    print("-" * 70)
    print(f"Final State: {format_statevector(initial_state.state)}")


# ==============================================================================
#                      Grover's Testing Utility
# ==============================================================================

def test_grover_search(n_qubits: int = 3, marked_index: int = 5):
    """
    Tests Grover's Algorithm on an n-qubit system.
    n_qubits: The number of qubits in the system (default 3).
    marked_index: The index of the target state to find (default 5, |101> for 3 qubits).
    """
    print("\n" + "=" * 70)

    # Check validity of marked_index for n_qubits
    N = 2 ** n_qubits
    if marked_index >= N:
        raise ValueError(f"marked_index ({marked_index}) is too large for {n_qubits} qubits (Max index {N-1}).")

    # Calculate binary string for printing
    marked_state_str = format(marked_index, f'0{n_qubits}b')
    print(f"RUNNING TEST: GROVER'S SEARCH ALGORITHM ({n_qubits} Qubits, Target |{marked_state_str}>)")

    # Calculate optimal iterations R and theoretical max probability P_max
    R = round(math.pi / 4 * math.sqrt(N))
    theta = math.asin(1 / math.sqrt(N))
    expected_prob = math.sin( (2 * R + 1) * theta ) ** 2

    # Build circ
    grover_qc = QuantumCircuit.grover_search(
        n_qubits=n_qubits,
        marked_state_index=marked_index
    )

    # Run Sim
    initial_state = QuantumState(n_qubits) # Starts at |0...0>
    print(f"Optimal Grover iterations (R): {R}")
    print(f"Applying {len(grover_qc.gates)} Gates to {n_qubits}-qubit state...")
    grover_qc.run(initial_state)

    probabilities = initial_state.get_probabilities()
    marked_prob = probabilities[marked_index]

    # The probability should be very close to the theoretical max.
    TOLERANCE = 0.02
    is_correct = np.isclose(marked_prob, expected_prob, atol=TOLERANCE)

    print(circuit_to_ascii(grover_qc))

    print(f"Target State: |{marked_state_str}> (Index {marked_index})")
    print(f"Theoretical Max Probability: {expected_prob:.4f}")
    print(f"Simulated Probability: {marked_prob:.4f}")
    print(f"Acceptance Range (P_max +/- {TOLERANCE}): [{expected_prob - TOLERANCE:.4f}, {expected_prob + TOLERANCE:.4f}]")
    print(f"Grover Test PASSED: {is_correct}")
    print("-" * 70)
    print(f"Final State: {format_statevector(initial_state.state)}")


# ==============================================================================
#                      QFT ADDER Testing Utility
# ==============================================================================

def test_qft_adder(A: int, B: int, n_bits: int, tolerance: float = 1e-6):
    """
    Generic QFT Adder test: verifies |B>|A> → |B + A mod 2^n>|A>

    Args:
        A (int): the addend (stored in lower n_bits)
        B (int): the target sum register (stored in upper n_bits)
        n_bits (int): number of bits in each register
    """

    if A >= 2**n_bits or B >= 2**n_bits:
        raise ValueError(f"A and B must be < 2^{n_bits}")

    print("="*70)
    print(f"Testing QFT Adder: A={A}, B={B}, n={n_bits} bits => Sum={(A+B)%(2**n_bits)}")
    print(f"Total Qubits = {2*n_bits}")

    # Encode |B>|A> (B in MSB register, A in LSB register)
    initial_index = (B << n_bits) + A

    # Expected final state index
    expected_sum = (A + B) % (2**n_bits)
    final_index = (expected_sum << n_bits) + A

    # Create initial state
    state = QuantumState(2*n_bits)
    state.basis_state(initial_index)

    print(f"Initial State index={initial_index} => |{B:0{n_bits}b}{A:0{n_bits}b}>")

    # Build adder
    qc = QuantumCircuit.qft_adder(n_bits)

    # Run
    qc.run(state)

    # Final statevector:
    sv = state.state

    # Check success condition:
    correct_prob = abs(sv[final_index])**2
    passed = correct_prob > (1 - tolerance)

    print(circuit_to_ascii(qc))

    if passed:
        print(f"PASSED: Correct result index {final_index} with prob={correct_prob:.4f}")
    else:
        print(f"FAILED: Expected index {final_index} but found prob={correct_prob:.4f}")

    print(f"Expected output state: |{expected_sum:0{n_bits}b}{A:0{n_bits}b}>")

    # Print non-zero amplitudes for inspection
    print("Final State Non-Zero Amplitudes:")
    for idx, amp in enumerate(sv):
        if abs(amp) > 1e-8:
            print(f"  idx {idx:2d}: |{idx:0{2*n_bits}b}> amplitude={amp}")

    print("="*70)


# ==============================================================================
#                              Main Test Suite
# ==============================================================================

def test_all_gates():
    """
    Main function to run all functional and visualization tests.
    """
    print("=" * 70)
    print("   QUANTUM GATE FUNCTIONAL AND BLOCH SPHERE VISUALIZATION TESTS")
    print("=" * 70)

    # Single-Qubit Standard Gates
    test_i_gate()
    test_x_gate()
    test_y_gate()
    test_z_gate()
    test_h_gate()
    test_s_gate()
    test_t_gate()

    # Single-Qubit Rotation Gates
    test_rx_pi_2()
    test_ry_pi_2()
    test_rz_pi()

    # Two-Qubit Standard/Swap Gates
    test_bell_state_cx()
    test_cz_gate()
    test_swap_gate()

    # Two-Qubit Controlled Rotation Gates
    test_crx_gate()
    test_cry_gate()
    test_crz_gate()

    # Multi-Controlled Gates (3 Qubits)
    test_mcx_gate()
    test_mcy_gate()
    test_mcz_gate()

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE: Check console output and Bloch Sphere plots for all gates.")
    print("=" * 70)


if __name__ == "__main__":
    test_all_gates()
    test_qft_decomposition(4, 7)
    test_qpe_s_gate()
    test_qpe_t_gate()
    test_qpe_approx_pi_3()
    test_grover_search(4, 13)