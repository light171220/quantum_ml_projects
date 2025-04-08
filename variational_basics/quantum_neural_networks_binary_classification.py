import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate a simple binary classification dataset (half-moons)
X, y = make_moons(n_samples=100, noise=0.3)
X = StandardScaler().fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# We'll use a simpler approach with fewer qubits
n_qubits = 2
n_layers = 2

# Create a quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Define the variational circuit
@qml.qnode(dev)
def circuit(inputs, weights):
    # Encode the inputs
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    
    # Variational layers
    for l in range(n_layers):
        # Rotations
        for i in range(n_qubits):
            qml.RX(weights[l][i][0], wires=i)
            qml.RY(weights[l][i][1], wires=i)
        
        # Entanglement
        qml.CNOT(wires=[0, 1])
    
    # Measurement
    return qml.expval(qml.PauliZ(0))

# Classifier function to convert to binary output
def classify(inputs, weights):
    return 1 if circuit(inputs, weights) > 0 else 0

# Define the cost function
def cost(weights, X, y):
    # Manual implementation to avoid autograd issues
    predictions = []
    for x in X:
        predictions.append((circuit(x, weights) + 1) / 2)  # Scale from [-1,1] to [0,1]
    
    # Manual MSE calculation
    loss = 0
    for pred, target in zip(predictions, y):
        loss += (pred - target) ** 2
    
    return loss / len(y)

# Initialize weights
weights = np.random.uniform(low=0, high=2*np.pi, size=(n_layers, n_qubits, 2))

# Simple training loop with manual gradient estimation (no autograd)
num_epochs = 30
step_size = 0.1
batch_size = 5
eps = 0.01  # Small value for finite difference gradient approximation
loss_history = []

print("Starting simplified QNN training...")

for epoch in range(num_epochs):
    # Select a random batch
    batch_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[batch_indices]
    y_batch = y_train[batch_indices]
    
    # Calculate current loss
    current_loss = cost(weights, X_batch, y_batch)
    loss_history.append(current_loss)
    
    # Manual gradient descent with finite differences
    grad = np.zeros_like(weights)
    
    # For each weight parameter
    for l in range(n_layers):
        for i in range(n_qubits):
            for j in range(2):
                # Create a copy of weights with a small perturbation
                weights_plus = weights.copy()
                weights_plus[l][i][j] += eps
                
                # Calculate the perturbed loss
                loss_plus = cost(weights_plus, X_batch, y_batch)
                
                # Estimate the gradient
                grad[l][i][j] = (loss_plus - current_loss) / eps
    
    # Update weights using gradient descent
    weights = weights - step_size * grad
    
    # Print progress
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {current_loss:.4f}")

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.savefig("qnn_training_loss.png")
plt.close()

# Evaluate on test set
test_predictions = [classify(x, weights) for x in X_test]
accuracy = accuracy_score(y_test, test_predictions)
print(f"\nTest accuracy: {accuracy:.4f}")

# Plot the test data and predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_predictions, cmap=plt.cm.coolwarm, 
            marker='o', edgecolors='k', s=80)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, 
            marker='x', s=120, alpha=0.5)
plt.title("QNN Predictions (o) vs True Labels (x)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("qnn_predictions.png")
plt.close()

# Visualize the quantum circuit
print("\nQuantum Neural Network Circuit:")
print(qml.draw(circuit)(X_train[0], weights))