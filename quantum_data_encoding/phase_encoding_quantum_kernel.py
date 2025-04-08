import pennylane as qml
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Generate dataset
centers = [[-0.5, -0.5], [0.5, 0.5]]
X, y = make_blobs(n_samples=200, centers=centers, cluster_std=0.3)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Split data
train_idx = np.random.choice(len(X), int(0.8*len(X)), replace=False)
test_idx = np.array(list(set(range(len(X))) - set(train_idx)))
X_train, y_train = X_scaled[train_idx], y[train_idx]
X_test, y_test = X_scaled[test_idx], y[test_idx]

# Define quantum device
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Define circuit with phase encoding
@qml.qnode(dev)
def phase_encoding_circuit(x1, x2):
    # Prepare initial state
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # Phase encoding
    qml.PhaseShift(x1, wires=0)
    qml.PhaseShift(x2, wires=1)
    
    # Entanglement
    qml.CNOT(wires=[0, 1])
    
    # More phase encodings
    qml.PhaseShift(x1 * x2, wires=0)  # Nonlinear feature
    
    # Measure all qubits in computational basis
    return qml.probs(wires=range(n_qubits))

# Define kernel function
def quantum_kernel(x1, x2):
    # Compute probabilities for each input
    probs1 = phase_encoding_circuit(x1[0], x1[1])
    probs2 = phase_encoding_circuit(x2[0], x2[1])
    
    # Kernel as inner product of probability vectors
    return np.sum(np.sqrt(probs1 * probs2))

# Create kernel matrix
def kernel_matrix(X1, X2):
    n_samples1 = X1.shape[0]
    n_samples2 = X2.shape[0]
    K = np.zeros((n_samples1, n_samples2))
    
    for i in range(n_samples1):
        for j in range(n_samples2):
            K[i, j] = quantum_kernel(X1[i], X2[j])
    
    return K

# Compute kernel matrices
K_train = kernel_matrix(X_train, X_train)
K_test = kernel_matrix(X_test, X_train)

# Train SVM with precomputed kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

# Predict
predictions = svm.predict(K_test)
accuracy = np.mean(predictions == y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Create a mesh grid for visualization
h = 0.02
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_scaled = scaler.transform(mesh_points)

# For visualization, use a subset of points
subset_idx = np.random.choice(len(mesh_points_scaled), 100)
subset_points = mesh_points_scaled[subset_idx]

# Compute kernel and predict
K_mesh = kernel_matrix(subset_points, X_train)
Z_subset = svm.predict(K_mesh)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.scatter(mesh_points[subset_idx, 0], mesh_points[subset_idx, 1], 
            c=Z_subset, cmap=plt.cm.coolwarm, alpha=0.3, s=20)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM with Quantum Kernel using Phase Encoding')
plt.tight_layout()
plt.savefig('phase_encoding_kernel.png', dpi=300, bbox_inches='tight')
plt.show()