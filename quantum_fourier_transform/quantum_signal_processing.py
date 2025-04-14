import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

n_qubits = 5
signal_length = 2**n_qubits

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

def encode_signal(signal, normalize=True):
    if normalize:
        norm = np.sqrt(np.sum(np.abs(signal)**2))
        if norm > 0:
            signal = signal / norm
    return signal

@qml.qnode(dev)
def quantum_signal_processing(signal_amplitudes):
    qml.AmplitudeEmbedding(features=signal_amplitudes, wires=range(n_qubits), normalize=True)
    qft(range(n_qubits))
    return qml.state()

t = np.linspace(0, 1, signal_length, endpoint=False)
signal = np.sin(2*np.pi*4*t) + 0.5*np.sin(2*np.pi*8*t)

signal_amplitudes = encode_signal(signal)

result = quantum_signal_processing(signal_amplitudes)

frequency_spectrum = np.abs(result)**2

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(np.arange(signal_length), frequency_spectrum)
plt.title("Frequency Spectrum (via QFT)")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.savefig("quantum_signal_processing.png")
plt.close()