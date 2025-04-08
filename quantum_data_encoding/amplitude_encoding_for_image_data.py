import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import normalize

# Load a small handwritten digit (8x8 image)
digits = datasets.load_digits()
sample_image = digits.images[0].reshape(-1)  # Flatten image
sample_label = digits.target[0]

# Normalize to create a valid quantum state (sum of squares = 1)
sample_image_normalized = normalize(sample_image.reshape(1, -1))[0]

# Number of qubits needed for amplitude encoding
num_qubits = int(np.ceil(np.log2(len(sample_image_normalized))))
dev = qml.device("default.qubit", wires=num_qubits)

# Prepare state using amplitude encoding
@qml.qnode(dev)
def amplitude_encoding_circuit():
    qml.AmplitudeEmbedding(sample_image_normalized, wires=range(num_qubits),
                          normalize=True, pad_with=0.0)
    # Return probabilities of each basis state
    return qml.probs(wires=range(num_qubits))

# Execute circuit
probabilities = amplitude_encoding_circuit()

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(digits.images[0], cmap='binary')
plt.title(f"Original Image (Digit {sample_label})")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(len(probabilities)), probabilities)
plt.xlabel("Basis State")
plt.ylabel("Probability")
plt.title("Quantum State Probabilities after Amplitude Encoding")
plt.tight_layout()

# Save the figure
plt.savefig('amplitude_encoding_image.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Number of classical values encoded: {len(sample_image_normalized)}")
print(f"Number of qubits used: {num_qubits}")