import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Use a moderate number of qubits for balance between accuracy and simulation time
n_counting = 6  # 6 counting qubits should give decent precision
n_target = 1
n_total = n_counting + n_target

dev = qml.device("default.qubit", wires=n_total)

def inverse_qft(wires):
    n_qubits = len(wires)
    
    for i in range(n_qubits//2):
        qml.SWAP(wires=[wires[i], wires[n_qubits-i-1]])
    
    for i in range(n_qubits-1, -1, -1):
        for j in range(n_qubits-1, i, -1):
            angle = -np.pi/2**(j-i)
            
            qml.CNOT(wires=[wires[j], wires[i]])
            qml.RZ(-angle/2, wires=wires[i])
            qml.CNOT(wires=[wires[j], wires[i]])
            qml.RZ(angle/2, wires=wires[i])
            
        qml.Hadamard(wires=wires[i])

@qml.qnode(dev)
def quantum_phase_estimation(phase):
    target_wire = n_counting
    
    # Initialize target qubit in the eigenstate of the unitary
    qml.PauliX(wires=target_wire)
    
    # Initialize counting register in superposition
    for i in range(n_counting):
        qml.Hadamard(wires=i)
    
    # Apply controlled unitaries
    # In this case, our unitary is a phase gate with phase = 2π*phase
    for i in range(n_counting):
        # Apply the controlled-U^(2^i) operation
        power = 2**i
        
        # Direct implementation of controlled phase gate
        qml.CNOT(wires=[i, target_wire])
        qml.PhaseShift(phase * np.pi * power, wires=target_wire)
        qml.CNOT(wires=[i, target_wire])
    
    # Apply inverse QFT to the counting register
    inverse_qft(range(n_counting))
    
    # Return the state
    return qml.state()

def extract_phase_estimate(state_vector, n_counting):
    # Calculate probabilities for the counting register states
    # We need to marginalize over the target qubit
    probabilities = np.zeros(2**n_counting)
    
    for i in range(2**n_counting):
        # Sum probabilities for each counting register state
        # (both when target is |0⟩ and |1⟩)
        probabilities[i] = np.abs(state_vector[i])**2 + np.abs(state_vector[i + 2**n_counting])**2
    
    # Find the most likely state
    most_likely_state = np.argmax(probabilities)
    
    # Convert to phase
    estimated_phase = most_likely_state / (2**n_counting)
    
    return estimated_phase

# Set the true phase - in this case 1/4
true_phase = 0.25

# Run the quantum phase estimation
result = quantum_phase_estimation(true_phase)

# Extract the estimated phase
estimated_phase = extract_phase_estimate(result, n_counting)

print(f"True phase: {true_phase}")
print(f"Estimated phase: {estimated_phase}")
print(f"Binary fraction: {int(estimated_phase * (2**n_counting))}/{2**n_counting}")
print(f"Absolute error: {abs(true_phase - estimated_phase)}")
print(f"Relative error: {abs(true_phase - estimated_phase)/true_phase*100:.6f}%")

# Calculate probabilities for all possible states of the counting register
full_probabilities = np.zeros(2**n_counting)
for i in range(2**n_counting):
    full_probabilities[i] = np.abs(result[i])**2 + np.abs(result[i + 2**n_counting])**2

# Find significant states for plotting
threshold = 0.01
significant_indices = [i for i, p in enumerate(full_probabilities) if p > threshold]
significant_probabilities = [full_probabilities[i] for i in significant_indices]

plt.figure(figsize=(12, 6))

if significant_indices:
    plt.bar([f"|{i:0{n_counting}b}⟩" for i in significant_indices], significant_probabilities)
    plt.title(f"QPE Probabilities for Phase = {true_phase}")
    plt.xlabel("Counting Register State")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
else:
    # If no states have significant probability, show the top 10 states
    top_indices = np.argsort(full_probabilities)[-10:]
    top_probabilities = [full_probabilities[i] for i in top_indices]
    
    plt.bar([f"|{i:0{n_counting}b}⟩" for i in top_indices], top_probabilities)
    plt.title(f"QPE Probabilities for Phase = {true_phase} (Top 10 states)")
    plt.xlabel("Counting Register State")
    plt.ylabel("Probability")
    plt.ylim(0, max(top_probabilities)*1.1)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("quantum_phase_estimation_corrected.png")
plt.close()