import pennylane as qml
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_moons(n_samples=100, noise=0.15)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Quantum device
dev = qml.device("default.qubit", wires=2)

# Define encoding methods
def angle_encoding(x, wires):
    qml.RX(x[0], wires=wires[0])
    qml.RY(x[1], wires=wires[1])

def iqp_encoding(x, wires):
    qml.RX(x[0], wires=wires[0])
    qml.RY(x[1], wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RZ(x[0] * x[1], wires=wires[0])  # Nonlinear term
    qml.RY(x[0] ** 2, wires=wires[1])   # Quadratic term

def dense_encoding(x, wires):
    qml.RX(x[0], wires=wires[0])
    qml.RY(x[1], wires=wires[1])
    qml.RZ(np.sin(x[0]), wires=wires[0])
    qml.RX(np.cos(x[1]), wires=wires[1])

# Define quantum circuits with different encodings
def quantum_model(encoding_function):
    @qml.qnode(dev)
    def circuit(x, weights):
        # Apply encoding
        encoding_function(x, wires=[0, 1])
        
        # Entanglement
        qml.CNOT(wires=[0, 1])
        
        # Trainable layers
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(weights[2], wires=0)
        qml.RY(weights[3], wires=1)
        
        # Measurement
        return qml.expval(qml.PauliZ(0))
    
    return circuit

# Cost function
def cost(weights, circuit, X, y):
    predictions = [circuit(x, weights) for x in X]
    return np.mean([(p - label) ** 2 for p, label in zip(predictions, y)])

# Train function
def train_model(encoding_function, X_train, y_train, steps=100):
    circuit = quantum_model(encoding_function)
    np.random.seed(42)
    weights = np.random.uniform(0, 2*np.pi, size=4)
    opt = qml.GradientDescentOptimizer(stepsize=0.3)
    
    costs = []
    for i in range(steps):
        weights = opt.step(lambda w: cost(w, circuit, X_train, y_train), weights)
        if (i+1) % 20 == 0:
            current_cost = cost(weights, circuit, X_train, y_train)
            costs.append(current_cost)
    
    return weights, circuit, costs

# Train models with different encodings
weights_angle, circuit_angle, costs_angle = train_model(angle_encoding, X_train, y_train)
weights_iqp, circuit_iqp, costs_iqp = train_model(iqp_encoding, X_train, y_train)
weights_dense, circuit_dense, costs_dense = train_model(dense_encoding, X_train, y_train)

# Evaluate models
def evaluate(weights, circuit, X, y):
    predictions = [circuit(x, weights) for x in X]
    binary_predictions = [1 if p > 0 else 0 for p in predictions]
    accuracy = np.mean([p == label for p, label in zip(binary_predictions, y)])
    return accuracy, binary_predictions

accuracy_angle, preds_angle = evaluate(weights_angle, circuit_angle, X_test, y_test)
accuracy_iqp, preds_iqp = evaluate(weights_iqp, circuit_iqp, X_test, y_test)
accuracy_dense, preds_dense = evaluate(weights_dense, circuit_dense, X_test, y_test)

print(f"Test Accuracy (Angle Encoding): {accuracy_angle:.4f}")
print(f"Test Accuracy (IQP Encoding): {accuracy_iqp:.4f}")
print(f"Test Accuracy (Dense Encoding): {accuracy_dense:.4f}")

# Create decision boundary plots
def plot_decision_boundary(weights, circuit, encoding_name):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([circuit(x, weights) for x in mesh_scaled])
    Z = np.array([1 if p > 0 else 0 for p in Z])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(f"Decision Boundary - {encoding_name}\nTest Accuracy: {evaluate(weights, circuit, X_test, y_test)[0]:.4f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plot_decision_boundary(weights_angle, circuit_angle, "Angle Encoding")
plt.subplot(1, 3, 2)
plot_decision_boundary(weights_iqp, circuit_iqp, "IQP Encoding")
plt.subplot(1, 3, 3)
plot_decision_boundary(weights_dense, circuit_dense, "Dense Encoding")
plt.tight_layout()
plt.savefig('encoding_comparison.png', dpi=300, bbox_inches='tight')
plt.show()