import pennylane as qml
import numpy as np

def qft(wires):
    n_qubits = len(wires)
    
    # Apply QFT circuit
    for i in range(n_qubits):
        qml.Hadamard(wires=wires[i])
        for j in range(i+1, n_qubits):
            # The controlled phase rotation
            qml.ControlledPhaseShift(np.pi/2**(j-i), control_wires=[wires[j]], wires=wires[i])
    
    # Swap qubits to match standard QFT output order
    for i in range(n_qubits//2):
        qml.SWAP(wires=[wires[i], wires[n_qubits-i-1]])

def inverse_qft(wires):
    n_qubits = len(wires)
    
    # Swap qubits first (reverse of QFT)
    for i in range(n_qubits//2):
        qml.SWAP(wires=[wires[i], wires[n_qubits-i-1]])
    
    # Apply inverse QFT circuit
    for i in range(n_qubits-1, -1, -1):
        for j in range(n_qubits-1, i, -1):
            # The controlled phase rotation with negative angle
            qml.ControlledPhaseShift(-np.pi/2**(j-i), control_wires=[wires[j]], wires=wires[i])
        qml.Hadamard(wires=wires[i])

# Create a quantum device
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

# Define a quantum circuit that prepares a state and applies QFT
@qml.qnode(dev)
def qft_circuit(basis_state):
    # Prepare the initial basis state
    for i in range(n_qubits):
        if basis_state[i]:
            qml.PauliX(wires=i)
    
    # Apply QFT
    qft(range(n_qubits))
    
    # Return the state vector
    return qml.state()

# Test the QFT with a basis state |101⟩
state = [1, 0, 1]  # |101⟩
result = qft_circuit(state)

# Display the result
print(f"QFT of |{''.join(map(str, state))}⟩:")
for i, amplitude in enumerate(result):
    # Only print non-zero amplitudes (accounting for numerical precision)
    if abs(amplitude) > 1e-10:
        print(f"|{i:0{n_qubits}b}⟩: {amplitude:.4f}")