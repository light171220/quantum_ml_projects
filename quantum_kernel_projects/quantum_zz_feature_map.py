import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate a non-linearly separable dataset (circles)
X, y = make_circles(n_samples=100, noise=0.1, factor=0.2, random_state=42)

# Scale the data to [0, 2π] range for encoding
scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define quantum device
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Define ZZ feature map (inspired by IBMQ's feature map)
@qml.qnode(dev)
def zz_feature_map(x, reps=2):
    # First layer of Hadamards
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # Repeated blocks
    for r in range(reps):
        # Data-embedding
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        
        # ZZ entanglement + non-linear transformation
        qml.CNOT(wires=[0, 1])
        qml.RZ(x[0] * x[1], wires=1)  # Non-linear ZZ interaction
        qml.CNOT(wires=[0, 1])
        
        # Second round of single-qubit rotations
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
            qml.Hadamard(wires=i)
    
    return qml.state()

# Function to compute the quantum kernel matrix using the ZZ feature map
def quantum_kernel_zz(x1, x2):
    state1 = zz_feature_map(x1)
    state2 = zz_feature_map(x2)
    kernel_value = np.abs(np.vdot(state1, state2))**2
    return kernel_value

# Generate the kernel matrix
def kernel_matrix(X1, X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    
    # Progress tracking for large matrices
    total = n1 * n2
    count = 0
    
    for i in range(n1):
        for j in range(n2):
            K[i, j] = quantum_kernel_zz(X1[i], X2[j])
            count += 1
            if count % 100 == 0 or count == total:
                print(f"Computed {count}/{total} kernel entries")
    
    return K

# Calculate the kernel matrices (this might take a few minutes)
print("Computing ZZ quantum kernel matrices...")
K_train = kernel_matrix(X_train, X_train)
K_test = kernel_matrix(X_test, X_train)

# Use precomputed kernel with SVM
print("Training SVM with ZZ quantum kernel...")
qsvm = SVC(kernel='precomputed')
qsvm.fit(K_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = qsvm.predict(K_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"ZZ quantum kernel classification accuracy: {accuracy:.4f}")

# Compare with classical RBF kernel for non-linear data
print("\nComparing with classical RBF kernel...")
classical_svm = SVC(kernel='rbf')
classical_svm.fit(X_train, y_train)
y_pred_classical = classical_svm.predict(X_test)
classical_accuracy = accuracy_score(y_test, y_pred_classical)
print(f"Classical RBF kernel accuracy: {classical_accuracy:.4f}")

# Visualize the decision boundaries
def plot_decision_boundary(model, kernel_type, X, y, is_quantum=False):
    h = 0.05  # Step size in mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    if is_quantum:
        # For quantum kernel, we need to calculate the kernel matrix
        X_mesh = np.c_[xx.ravel(), yy.ravel()]
        X_mesh_scaled = scaler.transform(X_mesh)
        K_mesh = kernel_matrix(X_mesh_scaled, X_train)
        Z = model.predict(K_mesh)
    else:
        # For classical kernel
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
    plt.title(f'Decision Boundary with {kernel_type}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(f'decision_boundary_{kernel_type.lower().replace(" ", "_")}.png')
    plt.close()

# Simplified example - just visualize the data
# (Full decision boundary visualization would take too long due to kernel calculations)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.title('Circles Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('circles_dataset.png')
plt.close()

print("\nZZ feature map circuit example:")
print(qml.draw(zz_feature_map)(X_train[0]))