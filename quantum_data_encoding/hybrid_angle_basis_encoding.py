import pennylane as qml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load iris dataset and select first two features
iris = load_iris()
X = iris.data[:, :2]  # Sepal length and width
y = iris.target

# Normalize continuous features to [0, π]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Create binary features (1 if above median, 0 otherwise)
binary_features = (X > np.median(X, axis=0)).astype(int)

# Define device with 4 qubits (2 for angle encoding, 2 for basis encoding)
dev = qml.device("default.qubit", wires=4)

# Define hybrid encoding circuit
@qml.qnode(dev)
def hybrid_encoding_circuit(continuous_features, binary_features, weights):
    # Angle encoding for continuous features
    qml.RX(continuous_features[0], wires=0)
    qml.RY(continuous_features[1], wires=1)
    
    # Basis encoding for binary features
    if binary_features[0] == 1:
        qml.PauliX(wires=2)
    if binary_features[1] == 1:
        qml.PauliX(wires=3)
    
    # Entanglement layer
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 3])
    
    # Parameterized rotation layer
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RZ(weights[2], wires=2)
    qml.RX(weights[3], wires=3)
    
    # Final entanglement
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    
    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Initialize weights
np.random.seed(42)
weights = np.random.uniform(0, 2*np.pi, size=4)

# Apply circuit to the first few samples
num_samples = 5
results = []
for i in range(num_samples):
    res = hybrid_encoding_circuit(X_scaled[i], binary_features[i], weights)
    results.append(res)
    print(f"Sample {i}: Class {y[i]}, Quantum Output: {res}")

# Visualize
results = np.array(results)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.scatter(X[:num_samples, 0], X[:num_samples, 1], 
           c='red', marker='x', s=100)
plt.title("Iris Dataset - First Two Features")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.subplot(1, 2, 2)
plt.imshow(results, cmap='RdBu', vmin=-1, vmax=1)
plt.title("Hybrid Encoding Circuit Output")
plt.xlabel("Qubit Index")
plt.ylabel("Sample Index")
plt.colorbar(label="Expectation <Z>")
plt.tight_layout()
plt.savefig('hybrid_encoding.png', dpi=300, bbox_inches='tight')
plt.show()