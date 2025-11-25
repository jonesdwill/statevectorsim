import numpy as np

from .quantum_state import QuantumState
from .quantum_gate import QuantumGate

class QuantumCircuit:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.gates: list[QuantumGate] = []

    def reset(self):
        """ Clears all gates from the circuit, returning to empty state."""
        self.gates = []

    def copy(self):
        """ Creates a new deep-copy QuantumCircuit instance """
        new_qc = QuantumCircuit(self.n)
        # Copy the list of references to the existing gate objects
        new_qc.gates = list(self.gates)
        return new_qc

    def add_gate(self, gate):
        """
        Add a single QuantumGate or a list/tuple of QuantumGate objects to circuit.
        e.g. qc.add_gate(QuantumGate.h([0, 1, 2]))
        """

        # list of gates
        if isinstance(gate, list) or isinstance(gate, tuple):
            self.gates.extend(gate)

        # single gate
        else:
            self.gates.append(gate)

    def run(self, quantum_state, inverse = False):
        """ Apply quantum gates to state in order """

        # forward
        if not inverse:
            for gate in self.gates:
                gate.apply(quantum_state)

        # inverse
        else:
            for gate in self.gates:
                gate.apply(quantum_state)

        return quantum_state

    def measure_all(self, initial_state: QuantumState, shots: int = 1024) -> dict[str, int]:
        """
        Run circuit multiple times and simulate measurement in the computational basis, returning a dictionary outcomes.

        Args:
            initial_state: The starting QuantumState for the simulation.
            shots: The number of times to run the simulation and measure.

        Returns:
            A dictionary where keys are measurement outcomes (e.g., '01')
            and values are the counts.
        """

        if initial_state.n != self.n:
            raise ValueError("Initial state must have the same number of qubits as the circuit.")

        results = {}

        for _ in range(shots):
            # create fresh copy of the initial state for each shot
            current_state = initial_state.copy()

            # run circuit
            self.run(current_state)

            # measure all qubits
            outcome_list = current_state.measure_all()

            # convert the list of bits into a measurement key (e.g., '01')
            outcome_str = "".join(map(str, outcome_list))

            # record the count
            results[outcome_str] = results.get(outcome_str, 0) + 1

        return results

    @staticmethod
    def bell():
        """
        Return 2-qubit Bell state circuit (|Φ+⟩ = (|00⟩ + |11⟩)/√2).
        """
        _qc = QuantumCircuit(2)
        _qc.add_gate(QuantumGate.h(0))  # Hadamard on qubit 0
        _qc.add_gate(QuantumGate.cx(0, 1))  # CNOT control=0, target=1
        return _qc

    @staticmethod
    def ghz(n_qubits=3):
        """
        Return a GHZ state circuit for n_qubits (|GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2). n >= 2.
        """
        if n_qubits < 2:
            raise ValueError("GHZ state requires at least 2 qubits.")

        _qc = QuantumCircuit(n_qubits)
        _qc.add_gate(QuantumGate.h(0))  # Hadamard on first qubit
        for t in range(1, n_qubits):
            _qc.add_gate(QuantumGate.cx(0, t))  # Entangle all other qubits with qubit 0

        return _qc
