import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Create a device with a single qubit
dev = qml.device("default.qubit", wires=1)

# Create a parameterized quantum circuit (variational circuit)
@qml.qnode(dev)
def rotation_circuit(params):
    # Apply rotation gates with parameters
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.RZ(params[2], wires=0)
    
    # Return the expectation value of Z
    return qml.expval(qml.PauliZ(0))

# Let's try different parameter values and observe the output
param_values = np.linspace(0, 2*np.pi, 20)
results = []

# Scan through different values for the X rotation while keeping Y and Z fixed
for theta in param_values:
    params = [theta, np.pi/4, np.pi/4]  # Only vary the first parameter
    results.append(rotation_circuit(params))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(param_values, results, 'b-')
plt.grid(True)
plt.xlabel('RX Rotation Angle (radians)')
plt.ylabel('Expectation Value <Z>')
plt.title('Effect of RX Parameter on Measurement')
plt.savefig('vqc_project1_plot.png')
plt.close()

print("Basic Parameterized Circuit Results:")
print(f"Circuit with parameters [0, π/4, π/4]: {rotation_circuit([0, np.pi/4, np.pi/4])}")
print(f"Circuit with parameters [π/2, π/4, π/4]: {rotation_circuit([np.pi/2, np.pi/4, np.pi/4])}")
print(f"Circuit with parameters [π, π/4, π/4]: {rotation_circuit([np.pi, np.pi/4, np.pi/4])}")

# Let's visualize the quantum state by calculating the expectation values
# of Pauli operators which gives us the Bloch vector components
def visualize_state(params):
    # Define circuits to measure each Pauli expectation
    @qml.qnode(dev)
    def measure_x(p):
        qml.RX(p[0], wires=0)
        qml.RY(p[1], wires=0)
        qml.RZ(p[2], wires=0)
        return qml.expval(qml.PauliX(0))
    
    @qml.qnode(dev)
    def measure_y(p):
        qml.RX(p[0], wires=0)
        qml.RY(p[1], wires=0)
        qml.RZ(p[2], wires=0)
        return qml.expval(qml.PauliY(0))
    
    @qml.qnode(dev)
    def measure_z(p):
        qml.RX(p[0], wires=0)
        qml.RY(p[1], wires=0)
        qml.RZ(p[2], wires=0)
        return qml.expval(qml.PauliZ(0))
    
    # Calculate the Bloch vector components
    x = measure_x(params)
    y = measure_y(params)
    z = measure_z(params)
    
    bloch_vector = [x, y, z]
    print(f"\nBloch vector for parameters {params}: {bloch_vector}")
    return bloch_vector

# Visualize for a few parameter sets
print("\nVisualizing quantum states:")
visualize_state([0, 0, 0])  # Initial state |0⟩
visualize_state([np.pi/2, np.pi/2, 0])  # Rotated state
visualize_state([np.pi, 0, 0])  # X gate (|1⟩)