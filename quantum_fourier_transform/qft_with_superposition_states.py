import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

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

@qml.qnode(dev)
def qft_superposition():
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    qft(range(n_qubits))
    
    return qml.state()

@qml.qnode(dev)
def qft_plus_state():
    qml.Hadamard(wires=0)
    
    qft(range(n_qubits))
    
    return qml.state()

result_superposition = qft_superposition()
result_plus = qft_plus_state()

def plot_significant_amplitudes(state_vector, title, filename):
    significant_indices = [i for i, amp in enumerate(state_vector) if abs(amp) > 1e-3]
    
    if not significant_indices:
        print(f"No significant amplitudes for {title}")
        return
    
    labels = [f"|{i:0{n_qubits}b}⟩" for i in significant_indices]
    probabilities = [np.abs(state_vector[i])**2 for i in significant_indices]
    phases = [np.angle(state_vector[i]) for i in significant_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(labels, probabilities)
    ax1.set_title(f"{title} - Probabilities")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, max(probabilities) * 1.1)
    
    ax2.bar(labels, phases)
    ax2.set_title(f"{title} - Phases")
    ax2.set_ylabel("Phase (radians)")
    ax2.set_ylim(-np.pi, np.pi)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_significant_amplitudes(result_superposition, "QFT of all qubits in |+⟩", "qft_all_superposition.png")
plot_significant_amplitudes(result_plus, "QFT of |+⟩|0⟩|0⟩|0⟩", "qft_plus_state.png")

print("QFT of superposition state (all qubits in |+⟩):")
for i, amplitude in enumerate(result_superposition):
    if abs(amplitude) > 1e-10:
        print(f"|{i:0{n_qubits}b}⟩: {amplitude:.4f}")

print("\nQFT of |+⟩|0⟩|0⟩|0⟩:")
for i, amplitude in enumerate(result_plus):
    if abs(amplitude) > 1e-10:
        print(f"|{i:0{n_qubits}b}⟩: {amplitude:.4f}")