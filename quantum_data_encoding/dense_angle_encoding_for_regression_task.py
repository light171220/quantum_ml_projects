import pennylane as qml
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
scaler_X = MinMaxScaler(feature_range=(0, 2*np.pi))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Define quantum device
dev = qml.device("default.qubit", wires=3)

# Define QML model with dense angle encoding
@qml.qnode(dev)
def dense_encoding_circuit(feature, weights):
    # Dense Angle Encoding (using multiple rotations for single feature)
    qml.RX(feature[0], wires=0)
    qml.RY(2*feature[0], wires=1)  # Double frequency
    qml.RZ(np.sin(feature[0]), wires=2)  # Nonlinear encoding
    
    # Entanglement
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    
    # Trainable rotation layer
    for i in range(3):
        qml.Rot(weights[i*3], weights[i*3+1], weights[i*3+2], wires=i)
    
    # Second entanglement
    qml.CNOT(wires=[2, 0])
    
    # Measurement
    return qml.expval(qml.PauliZ(0))

# Cost function
def cost(weights, X, y):
    predictions = [dense_encoding_circuit(x.reshape(-1), weights) for x in X]
    return np.mean((predictions - y) ** 2)

# Initialize weights
np.random.seed(42)
weights = np.random.uniform(0, 2*np.pi, size=9)

# Optimize
opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 150

costs = []
for i in range(steps):
    weights = opt.step(lambda w: cost(w, X_scaled, y_scaled), weights)
    current_cost = cost(weights, X_scaled, y_scaled)
    costs.append(current_cost)
    if (i+1) % 30 == 0:
        print(f"Step {i+1}, Cost: {current_cost:.4f}")

# Make predictions
predictions = [dense_encoding_circuit(x.reshape(-1), weights) for x in X_scaled]
predictions_original = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
y_original = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.title("Training Cost")
plt.xlabel("Optimization Step")
plt.ylabel("Cost")

plt.subplot(1, 2, 2)
sort_idx = np.argsort(X.flatten())
plt.scatter(X.flatten(), y, alpha=0.7)
plt.scatter(X.flatten(), predictions_original, alpha=0.7)
plt.plot(X.flatten()[sort_idx], predictions_original[sort_idx], 'r-')
plt.title("Quantum Regression with Dense Angle Encoding")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend(["Original data", "QML predictions", "QML function"])
plt.tight_layout()
plt.savefig('dense_angle_encoding_regression.png', dpi=300, bbox_inches='tight')
plt.show()