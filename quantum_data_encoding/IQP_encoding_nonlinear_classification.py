import pennylane as qml
import numpy as np
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate nonlinearly separable dataset
X, y = make_circles(n_samples=200, noise=0.1, factor=0.3)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Define quantum device
dev = qml.device("default.qubit", wires=2)

# Define QML model with IQP encoding
@qml.qnode(dev)
def iqp_circuit(features, weights):
    # First rotations
    qml.RX(features[0], wires=0)
    qml.RY(features[1], wires=1)
    
    # Entanglement
    qml.CNOT(wires=[0, 1])
    
    # Second layer of rotations (non-commuting)
    qml.RZ(features[0], wires=0)
    qml.RZ(features[1], wires=1)
    
    # Entanglement
    qml.CNOT(wires=[1, 0])
    
    # Trainable rotation layer
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RZ(weights[2], wires=0)
    qml.RX(weights[3], wires=1)
    
    # Measurement
    return qml.expval(qml.PauliZ(0))

# Cost function
def cost(weights, X, y):
    predictions = [iqp_circuit(x, weights) for x in X]
    return np.mean((predictions - y) ** 2)

# Initialize weights
np.random.seed(42)
weights = np.random.uniform(0, 2*np.pi, size=4)

# Optimize
opt = qml.GradientDescentOptimizer(stepsize=0.3)
steps = 100

for i in range(steps):
    weights = opt.step(lambda w: cost(w, X_scaled, y), weights)
    if (i+1) % 20 == 0:
        print(f"Step {i+1}, Cost: {cost(weights, X_scaled, y):.4f}")

# Make predictions
predictions = [iqp_circuit(x, weights) for x in X_scaled]
binary_predictions = [1 if p > 0 else 0 for p in predictions]

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=binary_predictions, cmap='coolwarm', 
            marker='x', s=100, alpha=0.5)
plt.title("Quantum Classification with IQP Encoding")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(["Original labels", "QML predictions"])
plt.savefig('iqp_encoding_classification.png', dpi=300, bbox_inches='tight')
plt.show()