import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

n_qubits = 3
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
def qft_with_input_state(input_state):
    for i in range(n_qubits):
        if input_state[i]:
            qml.PauliX(wires=i)
    
    qft(range(n_qubits))
    
    return qml.state()

def plot_state_vector(state_vector, title, filename):
    labels = [f"|{i:0{n_qubits}b}⟩" for i in range(2**n_qubits)]
    
    probabilities = np.abs(state_vector)**2
    phases = np.angle(state_vector)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(labels, probabilities)
    ax1.set_title(f"{title} - Probabilities")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, 1)
    
    ax2.bar(labels, phases)
    ax2.set_title(f"{title} - Phases")
    ax2.set_ylabel("Phase (radians)")
    ax2.set_ylim(-np.pi, np.pi)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

input_states = {
    "|0⟩": [0, 0, 0],
    "|1⟩": [0, 0, 1],
    "|5⟩": [1, 0, 1],
    "|7⟩": [1, 1, 1]
}

for name, state in input_states.items():
    result = qft_with_input_state(state)
    plot_state_vector(result, f"QFT of {name}", f"qft_state_{name.replace('|', '').replace('⟩', '')}.png")