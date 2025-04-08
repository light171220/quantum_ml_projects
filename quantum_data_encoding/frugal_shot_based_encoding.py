import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Define quantum device with shots
n_shots = 1000
dev_shots = qml.device("default.qubit", wires=2, shots=n_shots)

# Define circuit with angle encoding
@qml.qnode(dev_shots)
def shots_circuit(features, weights):
    qml.RX(features[0], wires=0)
    qml.RY(features[1], wires=1)
    qml.CNOT(wires=[0, 1])
    
    # Trainable weights
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    
    # Return samples instead of expectation
    return qml.sample(qml.PauliZ(0))

# Generate some test data
test_point = np.array([np.pi/4, np.pi/3])
np.random.seed(42)
weights = np.random.uniform(0, 2*np.pi, size=2)

# Run circuit with shots to get samples
samples = shots_circuit(test_point, weights)

# Calculate probability of measuring 1
prob_one = np.mean([1 if s == 1 else 0 for s in samples])
prob_zero = 1 - prob_one

# Visualize the shot distribution
plt.figure(figsize=(10, 6))
plt.bar(["-1", "+1"], [np.sum(samples == -1), np.sum(samples == 1)])
plt.title(f"Shot Distribution (n_shots={n_shots})")
plt.xlabel("Measurement Outcome")
plt.ylabel("Count")
plt.text(0, n_shots/2, f"Probability of +1: {prob_one:.4f}")
plt.text(0, n_shots/2 - 50, f"Probability of -1: {prob_zero:.4f}")
plt.ylim(0, n_shots)
plt.savefig('shot_based_encoding.png', dpi=300, bbox_inches='tight')
plt.show()

# Compare with analytical expectation
dev_analytic = qml.device("default.qubit", wires=2)

@qml.qnode(dev_analytic)
def analytic_circuit(features, weights):
    qml.RX(features[0], wires=0)
    qml.RY(features[1], wires=1)
    qml.CNOT(wires=[0, 1])
    
    # Trainable weights
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    
    return qml.expval(qml.PauliZ(0))

analytic_expectation = analytic_circuit(test_point, weights)
print(f"Analytical expectation: {analytic_expectation:.4f}")
print(f"Shot-based expectation: {np.mean(samples):.4f}")