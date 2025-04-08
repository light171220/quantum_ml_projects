import pennylane as qml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Load Iris dataset (only use 2 classes for binary classification)
iris = load_iris()
X = iris.data[:100, :2]  # Use only first two features and first two classes
y = iris.target[:100]    # Only use first two classes (0 and 1)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define quantum device
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Define a feature map circuit
@qml.qnode(dev)
def feature_map(x):
    # Encode the features into a quantum state
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    
    # Add some entanglement
    qml.CNOT(wires=[0, 1])
    
    # Return the state of the circuit
    return qml.state()

# Function to compute the quantum kernel
def quantum_kernel(x1, x2):
    # Calculate |<ψ(x1)|ψ(x2)>|^2
    state1 = feature_map(x1)
    state2 = feature_map(x2)
    
    # Calculate the inner product and take magnitude squared
    kernel_value = np.abs(np.vdot(state1, state2))**2
    return kernel_value

# Generate the kernel matrix for training data
def kernel_matrix(X1, X2):
    n_samples1 = X1.shape[0]
    n_samples2 = X2.shape[0]
    K = np.zeros((n_samples1, n_samples2))
    
    for i in range(n_samples1):
        for j in range(n_samples2):
            K[i, j] = quantum_kernel(X1[i], X2[j])
    
    return K

# Calculate the kernel matrices
print("Computing quantum kernel matrices...")
K_train = kernel_matrix(X_train, X_train)
K_test = kernel_matrix(X_test, X_train)

# Use precomputed kernel with SVM
print("Training SVM with quantum kernel...")
qsvm = SVC(kernel='precomputed')
qsvm.fit(K_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = qsvm.predict(K_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Quantum kernel classification accuracy: {accuracy:.4f}")

# Compare with classical RBF kernel
print("\nComparing with classical RBF kernel...")
classical_svm = SVC(kernel='rbf')
classical_svm.fit(X_train, y_train)
y_pred_classical = classical_svm.predict(X_test)
classical_accuracy = accuracy_score(y_test, y_pred_classical)
print(f"Classical RBF kernel accuracy: {classical_accuracy:.4f}")

# Print circuit for reference
print("\nQuantum feature map circuit:")
print(qml.draw(feature_map)(X_train[0]))