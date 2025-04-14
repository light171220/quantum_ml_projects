import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

def qft(wires):
    n_qubits = len(wires)
    
    for i in range(n_qubits):
        qml.Hadamard(wires=wires[i])
        for j in range(i+1, n_qubits):
            angle = np.pi/2**(j-i)
            
            qml.CNOT(wires=[wires[j], wires[i]])
            qml.RZ(-angle/2, wires=wires[i])
            qml.CNOT(wires=[wires[j], wires[i]])
            qml.RZ(angle/2, wires=wires[i])
    
    for i in range(n_qubits//2):
        qml.SWAP(wires=[wires[i], wires[n_qubits-i-1]])

def inverse_qft(wires):
    n_qubits = len(wires)
    
    for i in range(n_qubits//2):
        qml.SWAP(wires=[wires[i], wires[n_qubits-i-1]])
    
    for i in range(n_qubits-1, -1, -1):
        for j in range(n_qubits-1, i, -1):
            angle = -np.pi/2**(j-i)
            
            qml.CNOT(wires=[wires[j], wires[i]])
            qml.RZ(-angle/2, wires=wires[i])
            qml.CNOT(wires=[wires[j], wires[i]])
            qml.RZ(angle/2, wires=wires[i])
            
        qml.Hadamard(wires=wires[i])

n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qft_circuit(basis_state):
    for i in range(n_qubits):
        if basis_state[i]:
            qml.PauliX(wires=i)
    
    qft(range(n_qubits))
    
    return qml.state()

state = [1, 0, 1]  # |101⟩
result = qft_circuit(state)

print(f"QFT of |{''.join(map(str, state))}⟩:")
for i, amplitude in enumerate(result):
    if abs(amplitude) > 1e-10:
        print(f"|{i:0{n_qubits}b}⟩: {amplitude:.4f}")

# Visualize and save results
labels = [f"|{i:0{n_qubits}b}⟩" for i in range(2**n_qubits)]
probabilities = np.abs(result)**2
phases = np.angle(result)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(labels, probabilities)
ax1.set_title(f"QFT of |{''.join(map(str, state))}⟩ - Probabilities")
ax1.set_ylabel("Probability")
ax1.set_ylim(0, 1)

ax2.bar(labels, phases)
ax2.set_title(f"QFT of |{''.join(map(str, state))}⟩ - Phases")
ax2.set_ylabel("Phase (radians)")
ax2.set_ylim(-np.pi, np.pi)

plt.tight_layout()
plt.savefig("basic_qft_visualization.png")
plt.close()