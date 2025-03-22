import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load and prepare the Iris dataset
print("Loading and preparing the Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Display dataset information
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(target_names)}")
print(f"Class names: {target_names}")
print(f"Features: {feature_names}")
print(f"Class distribution: {np.bincount(y)}")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA to reduce to 2 dimensions for visualization and simpler encoding
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Define the number of qubits and quantum device
n_qubits = 4  # More qubits for more complex encoding
dev = qml.device("default.qubit", wires=n_qubits)

# 1. Define the Angle Embedding quantum circuit
@qml.qnode(dev)
def angle_embedding_circuit(features, weights):
    # Encode the 2 features using angle embedding (repeated to use all qubits)
    features_padded = np.pad(features, (0, n_qubits - len(features)), mode='wrap')
    qml.AngleEmbedding(features_padded, wires=range(n_qubits))
    
    # Apply trainable layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measure in computational basis - one for each class
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

# 2. Define the Amplitude Embedding quantum circuit
@qml.qnode(dev)
def amplitude_embedding_circuit(features, weights):
    # Normalize input features for amplitude embedding
    norm = np.linalg.norm(features)
    if norm == 0:
        normalized_features = features
    else:
        normalized_features = features / norm
    
    # Pad features to 2^n_qubits dimensions
    padding_size = 2**n_qubits - len(normalized_features)
    padded_features = np.pad(normalized_features, (0, padding_size), 'constant')
    
    # Normalize again after padding
    padded_features = padded_features / np.linalg.norm(padded_features)
    
    # Encode using amplitude embedding
    qml.AmplitudeEmbedding(padded_features, wires=range(n_qubits), normalize=True)
    
    # Apply trainable layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measure
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

# 3. Define a custom hybrid embedding circuit
@qml.qnode(dev)
def hybrid_embedding_circuit(features, weights):
    # First apply basis embedding on the first two qubits
    # Convert features to binary representation (0 or 1)
    binary_features = [1 if f > 0 else 0 for f in features]
    
    # Apply basis state encoding on first two qubits
    for i, bf in enumerate(binary_features):
        if bf == 1:
            qml.PauliX(i)
    
    # Then apply rotation encoding on all qubits
    for i in range(n_qubits):
        # Apply rotations based on feature values
        feature_idx = i % len(features)
        qml.RX(features[feature_idx], wires=i)
        qml.RY(features[feature_idx], wires=i)
    
    # Entangle qubits
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Apply trainable layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measure
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

# Define a classifier function to convert quantum outputs to class predictions
def classifier(x, weights, circuit):
    """Return class prediction from quantum circuit outputs"""
    outputs = circuit(x, weights)
    # Convert outputs to class probabilities (softmax-like normalization)
    return np.array(outputs)

# Define a one-hot encoding function
def one_hot(labels, num_classes=3):
    result = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        result[i, label] = 1
    return result

# Define cost function for training
def square_loss(predictions, targets):
    """Compute squared loss between predictions and targets"""
    loss = 0
    for pred, target in zip(predictions, targets):
        loss += np.sum((pred - target) ** 2)
    return loss / len(targets)

def cost(weights, features, targets, circuit):
    """Compute cost for all data points"""
    predictions = [classifier(x, weights, circuit) for x in features]
    return square_loss(predictions, targets)

# Define a training function for quantum circuits
def train_quantum_model(circuit, X_train, y_train, steps=100, batch_size=5):
    print(f"\nTraining quantum model with {circuit.__name__}...")
    
    # Initialize weights for the quantum model
    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits)
    weights = np.random.random(size=weight_shape)
    
    # Convert labels to one-hot encoding
    y_train_one_hot = one_hot(y_train)
    
    # Create optimizer
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    
    # Training loop
    costs = []
    for i in range(steps):
        # Mini-batch training
        batch_indices = np.random.randint(0, len(X_train), size=batch_size)
        X_batch = X_train[batch_indices]
        y_batch = y_train_one_hot[batch_indices]
        
        # Update weights
        weights = opt.step(lambda w: cost(w, X_batch, y_batch, circuit), weights)
        
        # Calculate cost for monitoring
        if i % 10 == 0:
            curr_cost = cost(weights, X_train, y_train_one_hot, circuit)
            costs.append(curr_cost)
            print(f"Step {i}: Cost = {curr_cost:.4f}")
    
    return weights, costs

# Function to evaluate the model
def evaluate_model(circuit, weights, X_test, y_test):
    """Evaluate model performance on test data"""
    print(f"\nEvaluating model with {circuit.__name__}...")
    
    # Get predictions for test data
    predictions = np.array([classifier(x, weights, circuit) for x in X_test])
    
    # Convert raw outputs to class labels
    pred_labels = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(pred_labels == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Calculate class-wise accuracy
    for i in range(3):
        class_idx = y_test == i
        if np.any(class_idx):
            class_acc = np.mean(pred_labels[class_idx] == y_test[class_idx])
            print(f"  Class {target_names[i]}: {class_acc:.4f}")
    
    return accuracy, pred_labels, predictions

# Function to visualize decision boundaries
def plot_decision_boundary(ax, weights, circuit, X, y, title):
    """Plot decision boundaries of the quantum classifier"""
    # Define mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                          np.arange(y_min, y_max, 0.02))
    
    # Predict for each point in mesh
    Z = np.zeros((xx.shape[0], xx.shape[1]))
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            features = np.array([xx[i, j], yy[i, j]])
            prediction = classifier(features, weights, circuit)
            Z[i, j] = np.argmax(prediction)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Plot training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    ax.set_xlabel('PCA Feature 1')
    ax.set_ylabel('PCA Feature 2')
    ax.set_title(title)
    
    # Add legend
    legend = ax.legend(scatter.legend_elements()[0], target_names,
                        loc="upper right", title="Classes")
    ax.add_artist(legend)

# Train and evaluate each quantum circuit
print("\n=== Training and Evaluating Quantum Models ===")

# Train and evaluate angle embedding circuit
angle_weights, angle_costs = train_quantum_model(
    angle_embedding_circuit, X_train, y_train, steps=100, batch_size=5
)
angle_acc, angle_preds, angle_raw = evaluate_model(
    angle_embedding_circuit, angle_weights, X_test, y_test
)

# Train and evaluate amplitude embedding circuit
amp_weights, amp_costs = train_quantum_model(
    amplitude_embedding_circuit, X_train, y_train, steps=100, batch_size=5
)
amp_acc, amp_preds, amp_raw = evaluate_model(
    amplitude_embedding_circuit, amp_weights, X_test, y_test
)

# Train and evaluate hybrid embedding circuit
hybrid_weights, hybrid_costs = train_quantum_model(
    hybrid_embedding_circuit, X_train, y_train, steps=100, batch_size=5
)
hybrid_acc, hybrid_preds, hybrid_raw = evaluate_model(
    hybrid_embedding_circuit, hybrid_weights, X_test, y_test
)

# Visualize results
print("\nVisualizing results...")
plt.figure(figsize=(16, 12))

# Plot training costs
plt.subplot(2, 2, 1)
plt.plot(range(0, 100, 10), angle_costs, 'o-', label='Angle Embedding')
plt.plot(range(0, 100, 10), amp_costs, 's-', label='Amplitude Embedding')
plt.plot(range(0, 100, 10), hybrid_costs, '^-', label='Hybrid Embedding')
plt.xlabel('Training Steps')
plt.ylabel('Cost')
plt.title('Training Cost Comparison')
plt.legend()
plt.grid(True)

# Plot decision boundaries
plt.subplot(2, 2, 2)
plot_decision_boundary(
    plt.gca(), angle_weights, angle_embedding_circuit, X_pca, y,
    f'Angle Embedding (Acc: {angle_acc:.2f})'
)

plt.subplot(2, 2, 3)
plot_decision_boundary(
    plt.gca(), amp_weights, amplitude_embedding_circuit, X_pca, y,
    f'Amplitude Embedding (Acc: {amp_acc:.2f})'
)

plt.subplot(2, 2, 4)
plot_decision_boundary(
    plt.gca(), hybrid_weights, hybrid_embedding_circuit, X_pca, y,
    f'Hybrid Embedding (Acc: {hybrid_acc:.2f})'
)

plt.tight_layout()
plt.savefig('quantum_iris_results.png')

# Visualize quantum circuits
print("\nVisualizing quantum circuits...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# Draw angle embedding circuit
angle_fig, _ = qml.draw_mpl(angle_embedding_circuit)(X_train[0], angle_weights)
angle_fig.suptitle("Angle Embedding Circuit", fontsize=16)
angle_fig.savefig("angle_embedding_circuit.png")

# Draw amplitude embedding circuit
amp_fig, _ = qml.draw_mpl(amplitude_embedding_circuit)(X_train[0], amp_weights)
amp_fig.suptitle("Amplitude Embedding Circuit", fontsize=16)
amp_fig.savefig("amplitude_embedding_circuit.png")

# Draw hybrid embedding circuit
hybrid_fig, _ = qml.draw_mpl(hybrid_embedding_circuit)(X_train[0], hybrid_weights)
hybrid_fig.suptitle("Hybrid Embedding Circuit", fontsize=16)
hybrid_fig.savefig("hybrid_embedding_circuit.png")

print("\n=== Quantum ML Project on Iris Dataset Completed ===")
print(f"Best model: {max((angle_acc, 'Angle Embedding'), (amp_acc, 'Amplitude Embedding'), (hybrid_acc, 'Hybrid Embedding'), key=lambda x: x[0])[1]}")
print("Results saved as 'quantum_iris_results.png'")
print("Quantum circuits visualized and saved as separate PNG files")