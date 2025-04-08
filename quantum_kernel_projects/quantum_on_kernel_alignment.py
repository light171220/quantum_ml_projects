import pennylane as qml
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

# Set random seed for reproducibility
np.random.seed(42)

print("Loading and preprocessing data...")
# Load and prepare data
data = load_breast_cancer()
X = data.data
y = data.target

# Feature selection - use only 2 features for simplicity and speed
# Select the two most important features
feature_indices = [0, 7]  # Mean radius and mean concave points
X = X[:, feature_indices]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data - use a smaller subset for faster computation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further subsample the training data for faster computation
n_train_samples = min(100, len(X_train))  # Limit to 100 samples max
indices = np.random.choice(len(X_train), n_train_samples, replace=False)
X_train_small = X_train[indices]
y_train_small = y_train[indices]

# Define quantum device - use just 2 qubits for the 2 features
n_qubits = 2  # One qubit per feature
dev = qml.device("default.qubit", wires=n_qubits)

print(f"Using {n_qubits} qubits for {X.shape[1]} features")
print(f"Training with {n_train_samples} samples (out of {len(X_train)} total)")

# Define parameterized quantum feature map (simplified)
@qml.qnode(dev)
def parameterized_feature_map(x, params):
    # First layer - feature embedding
    for i in range(n_qubits):
        qml.RX(x[i], wires=i)
    
    # Entangling layer
    qml.CNOT(wires=[0, 1])
    
    # Parameterized rotation layer
    for i in range(n_qubits):
        qml.RZ(params[0, i], wires=i)
        qml.RY(params[1, i], wires=i)
    
    # Second entangling layer
    qml.CNOT(wires=[0, 1])
    
    return qml.state()

# Function to compute quantum kernel with parameters
def quantum_kernel_param(x1, x2, params):
    state1 = parameterized_feature_map(x1, params)
    state2 = parameterized_feature_map(x2, params)
    kernel_value = np.abs(np.vdot(state1, state2))**2
    return kernel_value

# Function to compute the kernel matrix with progress tracking
def kernel_matrix_param(X1, X2, params):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    
    start_time = time.time()
    total_pairs = n1 * n2
    
    for i in range(n1):
        if i % max(1, n1 // 10) == 0:  # Show progress every 10%
            elapsed = time.time() - start_time
            progress = i * n2 / total_pairs
            if progress > 0:
                eta = elapsed / progress * (1 - progress)
                print(f"Computing kernel matrix: {progress*100:.1f}% complete, ETA: {eta:.1f}s")
            
        for j in range(n2):
            K[i, j] = quantum_kernel_param(X1[i], X2[j], params)
    
    print(f"Kernel matrix computation completed in {time.time() - start_time:.1f}s")
    return K

# Kernel alignment cost function
def kernel_alignment(K, y):
    """Calculate alignment between kernel matrix K and target kernel from labels y"""
    # Construct target kernel: y_i * y_j (ideal kernel)
    y_matrix = np.outer(y, y)
    
    # Center the kernel matrix
    K_centered = K - np.mean(K, axis=0)
    
    # Calculate Frobenius inner product <K, y_matrix>
    alignment_num = np.sum(K_centered * y_matrix)
    
    # Calculate normalization terms
    K_norm = np.sqrt(np.sum(K_centered * K_centered))
    y_norm = np.sqrt(np.sum(y_matrix * y_matrix))
    
    # Return negative alignment (for minimization)
    if K_norm == 0 or y_norm == 0:
        return 0  # Avoid division by zero
    
    alignment = alignment_num / (K_norm * y_norm)
    return -alignment  # Negative for minimization

# Generate an even smaller subset for kernel parameter optimization
n_samples_subset = min(30, n_train_samples)
indices = np.random.choice(n_train_samples, n_samples_subset, replace=False)
X_opt_subset = X_train_small[indices]
y_opt_subset = y_train_small[indices]

# Initialize parameters
np.random.seed(42)
initial_params = np.random.uniform(-np.pi, np.pi, (2, n_qubits))  # Simplified parameter structure

print(f"Optimizing kernel parameters using {n_samples_subset} samples...")

# Generate more diverse parameter options
param_options = []
alignment_scores = []

# Define some parameter variations to try
n_variations = 5
for i in range(n_variations):
    # Create parameter variations with more diversity
    if i == 0:
        # Use initial parameters
        param_option = initial_params.copy()
    elif i == 1:
        # Scale up
        param_option = initial_params * 2.0
    elif i == 2:
        # Scale down
        param_option = initial_params * 0.5
    elif i == 3:
        # Shift phase
        param_option = initial_params + np.pi/2
    else:
        # Random variation
        param_option = np.random.uniform(-np.pi, np.pi, (2, n_qubits))
    
    param_options.append(param_option)
    
    # Calculate kernel and alignment
    K = kernel_matrix_param(X_opt_subset, X_opt_subset, param_option)
    alignment = kernel_alignment(K, y_opt_subset)
    alignment_scores.append(alignment)
    print(f"Parameter set {i+1}/{n_variations}, Alignment score: {-alignment:.4f}")

# Select best parameters
best_idx = np.argmin(alignment_scores)
best_params = param_options[best_idx]
print(f"\nBest alignment score: {-alignment_scores[best_idx]:.4f}")

# Use a very small subset for testing the classifier (for demo purposes)
n_test_samples = min(40, len(X_test))
test_indices = np.random.choice(len(X_test), n_test_samples, replace=False)
X_test_small = X_test[test_indices]
y_test_small = y_test[test_indices]

print("\nComputing kernel matrices for classification...")
# Compute final kernel matrices with best parameters (using smaller subsets)
K_train_final = kernel_matrix_param(X_train_small, X_train_small, best_params)
K_test_final = kernel_matrix_param(X_test_small, X_train_small, best_params)

# Train SVM with optimized kernel
print("Training SVM with optimized quantum kernel...")
qsvm = SVC(kernel='precomputed')
qsvm.fit(K_train_final, y_train_small)

# Make predictions
print("Making predictions...")
y_pred = qsvm.predict(K_test_final)

# Calculate accuracy
accuracy = accuracy_score(y_test_small, y_pred)
print(f"Optimized quantum kernel classification accuracy: {accuracy:.4f}")

# Compare with classical SVM
print("\nComparing with classical RBF kernel...")
classical_svm = SVC(kernel='rbf')
classical_svm.fit(X_train_small, y_train_small)
y_pred_classical = classical_svm.predict(X_test_small)
classical_accuracy = accuracy_score(y_test_small, y_pred_classical)
print(f"Classical RBF kernel accuracy: {classical_accuracy:.4f}")

# Plot alignment scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_variations+1), [-score for score in alignment_scores], marker='o')
plt.grid(True)
plt.xlabel('Parameter Set')
plt.ylabel('Kernel Alignment Score (higher is better)')
plt.title('Kernel Alignment Optimization')
plt.savefig('kernel_alignment_optimization.png')
plt.close()

print("\nParameterized feature map circuit example:")
print(qml.draw(parameterized_feature_map)(X_train_small[0], best_params))
print("\nOptimization visualization saved as 'kernel_alignment_optimization.png'")