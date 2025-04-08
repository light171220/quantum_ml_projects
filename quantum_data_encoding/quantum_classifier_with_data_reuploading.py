import pennylane as qml
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and prepare data
iris = load_iris()
X = iris.data[:, :2]  # Use only first two features for visualization
y = iris.target

# Normalize features to [0, 2π]
scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
X_scaled = scaler.fit_transform(X)

# Binarize targets for binary classification (setosa vs. rest)
y_binary = np.where(y == 0, 0, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.3, random_state=42)

# Define quantum device
dev = qml.device("default.qubit", wires=2)

# Define QML model with data re-uploading
@qml.qnode(dev)
def data_reuploading_circuit(x, weights):
    # Number of layers
    n_layers = 3
    
    # Iterate through layers
    for layer in range(n_layers):
        # Encoding block
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)
        
        # Entanglement
        qml.CNOT(wires=[0, 1])
        
        # Trainable rotation layer
        qml.RX(weights[layer*4], wires=0)
        qml.RY(weights[layer*4 + 1], wires=1)
        qml.RZ(weights[layer*4 + 2], wires=0)
        qml.RZ(weights[layer*4 + 3], wires=1)
    
    # Measurement
    return qml.expval(qml.PauliZ(0))

# Cost function
def cost(weights, X, y):
    predictions = [data_reuploading_circuit(x, weights) for x in X]
    # Map predictions from [-1,1] to [0,1]
    predictions_binary = [(p + 1) / 2 for p in predictions]
    return np.mean((predictions_binary - y) ** 2)

# Initialize weights
np.random.seed(42)
n_params = 12  # 4 params per layer * 3 layers
weights = np.random.uniform(0, 2*np.pi, size=n_params)

# Train the model
opt = qml.GradientDescentOptimizer(stepsize=0.2)
steps = 200
batch_size = 10

loss_history = []

for i in range(steps):
    # Mini-batch gradient descent
    batch_indices = np.random.choice(len(X_train), batch_size, replace=False)
    X_batch = X_train[batch_indices]
    y_batch = y_train[batch_indices]
    
    weights = opt.step(lambda w: cost(w, X_batch, y_batch), weights)
    
    if (i+1) % 20 == 0:
        loss = cost(weights, X_train, y_train)
        loss_history.append(loss)
        print(f"Step {i+1}, Loss: {loss:.4f}")

# Evaluate on test set
test_predictions = [(data_reuploading_circuit(x, weights) + 1) / 2 for x in X_test]
test_binary = [1 if p > 0.5 else 0 for p in test_predictions]
test_accuracy = np.mean(test_binary == y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Create decision boundary
def plot_decision_boundary():
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_scaled = scaler.transform(mesh_points)
    
    # For a large mesh, predict on a subset to speed up
    subset_idx = np.random.choice(len(mesh_scaled), 500)
    subset_points = mesh_scaled[subset_idx]
    
    Z = [(data_reuploading_circuit(x, weights) + 1) / 2 for x in subset_points]
    Z_binary = [1 if p > 0.5 else 0 for p in Z]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_binary, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.scatter(mesh_points[subset_idx, 0], mesh_points[subset_idx, 1], 
               c=Z_binary, cmap=plt.cm.coolwarm, alpha=0.3, s=20)
    plt.title(f"Decision Boundary with Data Re-uploading\nTest Accuracy: {test_accuracy:.4f}")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(20, steps+1, 20), loss_history)
    plt.title("Training Loss")
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data_reuploading_project.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_decision_boundary()