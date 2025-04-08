import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Create binary feature vectors
binary_data = np.array([
    [0, 0, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 1]
])

# Define device
dev = qml.device("default.qubit", wires=4)

# Define QML circuit with basis encoding
@qml.qnode(dev)
def basis_encoding_circuit(binary_vector):
    # Basis encoding - flip qubit if feature is 1
    for i, bit in enumerate(binary_vector):
        if bit == 1:
            qml.PauliX(wires=i)
    
    # Apply some entangling operations
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    
    # Measure all qubits in Z basis
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Apply circuit to each data point
results = np.array([basis_encoding_circuit(x) for x in binary_data])

# Visualization
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(binary_data, cmap='binary')
plt.title("Original Binary Data")
plt.xlabel("Feature Index")
plt.ylabel("Sample Index")

plt.subplot(1, 2, 2)
plt.imshow(results, cmap='RdBu', vmin=-1, vmax=1)
plt.title("Quantum Circuit Output")
plt.xlabel("Qubit Index")
plt.ylabel("Sample Index")
plt.colorbar(label="Expectation <Z>")
plt.tight_layout()

# Save the figure
plt.savefig('basis_encoding_binary.png', dpi=300, bbox_inches='tight')
plt.show()