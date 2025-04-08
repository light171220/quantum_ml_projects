import pennylane as qml
import numpy as np

n_qubits = 3

dev = qml.device("default.qubit", wires=n_qubits)

def constant_0_oracle(wires):
    pass

def constant_1_oracle(wires):
    qml.PauliX(wires=wires[-1])
    
def balanced_oracle_1(wires):
    qml.CNOT(wires=[wires[0], wires[-1]])
    
def balanced_oracle_2(wires):
    for i in range(n_qubits - 1):
        if i % 2 == 0:
            qml.CNOT(wires=[wires[i], wires[-1]])

@qml.qnode(dev)
def deutsch_jozsa(oracle):
    for i in range(n_qubits - 1):
        qml.Hadamard(wires=i)
    
    qml.PauliX(wires=n_qubits - 1)
    qml.Hadamard(wires=n_qubits - 1)
    
    oracle(range(n_qubits))
    
    for i in range(n_qubits - 1):
        qml.Hadamard(wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits - 1)]

def is_constant(oracle):
    results = deutsch_jozsa(oracle)
    
    all_ones = True
    for result in results:
        if result < 0.99:  # Check if expectation value is close to 1
            all_ones = False
            break
    
    return all_ones

print("Testing Deutsch-Jozsa algorithm with different oracles:")
print(f"Constant oracle (always 0): {'Constant' if is_constant(constant_0_oracle) else 'Balanced'}")
print(f"Constant oracle (always 1): {'Constant' if is_constant(constant_1_oracle) else 'Balanced'}")
print(f"Balanced oracle (example 1): {'Constant' if is_constant(balanced_oracle_1) else 'Balanced'}")
print(f"Balanced oracle (example 2): {'Constant' if is_constant(balanced_oracle_2) else 'Balanced'}")

print("\nQuantum circuit for the Deutsch-Jozsa algorithm with balanced_oracle_1:")
circuit = qml.tape.QuantumTape()
with circuit:
    for i in range(n_qubits - 1):
        qml.Hadamard(wires=i)
    
    qml.PauliX(wires=n_qubits - 1)
    qml.Hadamard(wires=n_qubits - 1)
    
    balanced_oracle_1(range(n_qubits))
    
    for i in range(n_qubits - 1):
        qml.Hadamard(wires=i)

print(qml.draw(circuit))