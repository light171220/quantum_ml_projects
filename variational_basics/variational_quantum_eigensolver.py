import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Create a device with 2 qubits
dev = qml.device("default.qubit", wires=2)

# Define our ansatz (parameterized trial wavefunction)
def ansatz(params, wires):
    # Entangled state preparation
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    
    # More rotations
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])

# Define the Hamiltonian for a simple Heisenberg model
coeffs = [1.0, 1.0, 1.0]
observables = [
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliY(0) @ qml.PauliY(1),
    qml.PauliZ(0) @ qml.PauliZ(1)
]
hamiltonian = qml.Hamiltonian(coeffs, observables)

# Create the VQE cost function (expected energy)
@qml.qnode(dev)
def cost_function(params):
    ansatz(params, wires=[0, 1])
    return qml.expval(hamiltonian)

# Define a gradient-based optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# Initialize random parameters
np.random.seed(42)
initial_params = np.random.uniform(0, 2*np.pi, size=4)

# Track optimization progress
params = initial_params
energy_history = []

# Run the optimization
max_iterations = 100

print("Starting VQE optimization...")
for i in range(max_iterations):
    # Explicitly mark parameters as trainable
    params_tensor = qml.numpy.array(params, requires_grad=True)
    
    # Update the parameters
    params, prev_energy = opt.step_and_cost(cost_function, params_tensor)
    
    # Store current energy for plotting
    energy_history.append(prev_energy)
    
    # Print progress
    if (i + 1) % 10 == 0:
        print(f"Iteration {i + 1}: Energy = {prev_energy:.6f}")

# Plot the optimization progress
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_iterations + 1), energy_history, 'b-')
plt.grid(True)
plt.xlabel('Optimization Iteration')
plt.ylabel('Energy')
plt.title('VQE Optimization Progress')
plt.savefig('vqe_optimization.png')
plt.close()

# Print final results
print("\nVQE Results:")
print(f"Final parameters: {params}")
print(f"Final energy: {energy_history[-1]:.6f}")

# For comparison: calculate the exact ground state energy
# For the Heisenberg model with coupling = 1, it's -3
print(f"Exact ground state energy: {-3:.6f}")
print(f"Optimization error: {abs(-3 - energy_history[-1]):.6f}")

# Visualize the quantum circuit
print("\nQuantum Circuit for the VQE Ansatz:")
circuit = qml.tape.QuantumTape()
with circuit:
    ansatz(params, wires=[0, 1])
print(qml.draw(circuit))

# Check the final quantum state
@qml.qnode(dev)
def final_state():
    ansatz(params, wires=[0, 1])
    return qml.state()

state = final_state()
print("\nFinal quantum state (amplitudes):")
for i, amplitude in enumerate(state):
    base = format(i, '02b')
    print(f"|{base}⟩: {amplitude.real:.4f}{'+' if amplitude.imag >= 0 else ''}{amplitude.imag:.4f}j")