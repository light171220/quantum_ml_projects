import pennylane as qml
import numpy as np
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_circles(n_samples=100, noise=0.1, factor=0.3)
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_scaled = scaler.fit_transform(X)

# Define quantum device
dev = qml.device("default.qubit", wires=4)

# Generate random Fourier features frequencies
np.random.seed(42)
n_features = 4
omega = np.random.normal(0, 1, size=(2, n_features))

@qml.qnode(dev)
def fourier_encoding_circuit(x, weights):
    # Calculate Fourier features
    fourier_features = np.concatenate([
        np.cos(np.dot(x, omega)),
        np.sin(np.dot(x, omega))
    ])
    
    # Encode Fourier features using angle encoding
    for i in range(4):
        qml.RY(fourier_features[i], wires=i)
    
    # Entanglement layer
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    
    # Trainable rotation layer
    for i in range(4):
        qml.RX(weights[i], wires=i)
    
    # Return expectation
    return qml.expval(qml.PauliZ(0))

# Initialize weights
weights = np.random.uniform(0, 2*np.pi, size=4)

# Visualize the Fourier feature transformation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Transform data to Fourier features
fourier_transform = np.zeros((len(X_scaled), 8))
for i, x in enumerate(X_scaled):
    fourier_transform[i] = np.concatenate([
        np.cos(np.dot(x, omega)),
        np.sin(np.dot(x, omega))
    ])

# Visualize first two dimensions of Fourier features
plt.subplot(1, 2, 2)
plt.scatter(fourier_transform[:, 0], fourier_transform[:, 1], c=y, cmap='viridis')
plt.title("First Two Fourier Features")
plt.xlabel("Cos Feature 1")
plt.ylabel("Cos Feature 2")
plt.tight_layout()
plt.savefig('random_fourier_features_encoding.png', dpi=300, bbox_inches='tight')
plt.show()

# Evaluate quantum circuit for some points
results = []
for x in X_scaled[:10]:
    result = fourier_encoding_circuit(x, weights)
    results.append(result)

print(f"Circuit outputs for first 10 points: {results}")