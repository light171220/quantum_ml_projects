import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Define a hardware efficient circuit with restricted gates
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def hardware_efficient_circuit(features, weights):
    # Initial state preparation
    for i in range(3):
        qml.Hadamard(wires=i)
    
    # First encoding layer (only using RZ gates, which are native to many platforms)
    for i in range(3):
        qml.RZ(features[i % len(features)], wires=i)
    
    # Entanglement layer using only nearest-neighbor connectivity
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    
    # Trainable rotation layer (hardware efficient)
    for i in range(3):
        qml.RZ(weights[i], wires=i)
    
    # Second entanglement layer
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    
    # Final rotations
    for i in range(3):
        qml.RZ(weights[i+3], wires=i)
    
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

# Generate random data
np.random.seed(42)
features = np.random.uniform(0, np.pi, size=2)
weights = np.random.uniform(0, 2*np.pi, size=6)

# Evaluate circuit
result = hardware_efficient_circuit(features, weights)
print(f"Circuit output: {result}")

# Analyze parameter sensitivity
param_range = np.linspace(0, 2*np.pi, 50)
weight_outputs = []

for param in param_range:
    # Modify just one weight parameter
    modified_weights = weights.copy()
    modified_weights[0] = param
    output = hardware_efficient_circuit(features, modified_weights)
    weight_outputs.append(output[0])  # Track first qubit output

# Plot parameter sensitivity
plt.figure(figsize=(10, 6))
plt.plot(param_range, weight_outputs)
plt.title("Hardware-Efficient Encoding: Parameter Sensitivity")
plt.xlabel("Weight Parameter Value")
plt.ylabel("Circuit Output (First Qubit)")
plt.grid(True)
plt.savefig('hardware_efficient_encoding.png', dpi=300, bbox_inches='tight')
plt.show()