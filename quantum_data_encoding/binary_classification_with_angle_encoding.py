import pennylane as qml
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate binary classification dataset
X, y = make_moons(n_samples=200, noise=0.1)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Define quantum device
dev = qml.device("default.qubit", wires=2)

# Define QML model with angle encoding
@qml.qnode(dev)
def quantum_circuit(features, weights):
    # Angle encoding
    qml.RX(features[0], wires=0)
    qml.RY(features[1], wires=1)
    
    # Entanglement
    qml.CNOT(wires=[0, 1])
    
    # Trainable rotation layer
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    
    # Measurement
    return qml.expval(qml.PauliZ(0))

# Define a simple cost function
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    return loss / len(labels)

# Define a cost function that uses the square loss
def cost(weights):
    predictions = [quantum_circuit(x, weights) for x in X_scaled]
    return square_loss(y, predictions)

# Initialize weights with requires_grad=True
np.random.seed(42)
init_weights = np.random.uniform(0, 2*np.pi, size=2)
weights = qml.numpy.array(init_weights, requires_grad=True)

# Optimize
opt = qml.GradientDescentOptimizer(stepsize=0.5)
steps = 100
cost_history = []

for i in range(steps):
    # Apply optimization step
    weights = opt.step(cost, weights)
    
    # Calculate cost for logging
    current_cost = cost(weights)
    cost_history.append(current_cost)
    
    if (i+1) % 10 == 0:
        print(f"Step {i+1}, Cost: {current_cost:.4f}")

# Plot the cost history
plt.figure(figsize=(8, 5))
plt.plot(range(1, steps+1), cost_history)
plt.title("Training Cost History")
plt.xlabel("Optimization Step")
plt.ylabel("Cost")
plt.grid(True)
plt.savefig('angle_encoding_training_cost.png', dpi=300, bbox_inches='tight')
plt.show()

# Make predictions
predictions = [quantum_circuit(x, weights) for x in X_scaled]
binary_predictions = [1 if p > 0 else 0 for p in predictions]

# Calculate accuracy
accuracy = np.mean([pred == label for pred, label in zip(binary_predictions, y)])
print(f"Classification accuracy: {accuracy:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=binary_predictions, cmap='coolwarm', 
            marker='x', s=100, alpha=0.5)
plt.title(f"Quantum Binary Classification with Angle Encoding\nAccuracy: {accuracy:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(["Original labels", "QML predictions"])
plt.savefig('angle_encoding_classification_fixed.png', dpi=300, bbox_inches='tight')
plt.show()