# Variational Quantum Circuits Projects

This repository contains three progressive projects about variational quantum circuits (VQCs) using PennyLane. These projects demonstrate fundamental concepts in quantum machine learning, from basic parameterized circuits to quantum neural networks.

## Table of Contents
1. [Introduction to Variational Quantum Circuits](#introduction)
2. [Project 1: Basic Parameterized Rotation Circuit](#project-1-basic-parameterized-rotation-circuit)
3. [Project 2: Variational Quantum Eigensolver (VQE)](#project-2-variational-quantum-eigensolver-vqe)
4. [Project 3: Quantum Neural Network](#project-3-quantum-neural-network)
5. [Requirements](#requirements)

## Introduction

Variational Quantum Circuits are a cornerstone of quantum machine learning, leveraging classical optimization to train quantum circuits for various tasks. These circuits contain parameters that can be adjusted to optimize performance for specific applications.

## Project 1: Basic Parameterized Rotation Circuit

File: `basic_parameterized_rotation.py`

### Description
This introductory project demonstrates how parameters affect a quantum state using single-qubit rotation gates.

### Code Breakdown

```python
# Device setup
dev = qml.device("default.qubit", wires=1)  # Create a single-qubit quantum device

# Parameterized quantum circuit definition
@qml.qnode(dev)
def rotation_circuit(params):
    qml.RX(params[0], wires=0)  # X-rotation with parameter
    qml.RY(params[1], wires=0)  # Y-rotation with parameter
    qml.RZ(params[2], wires=0)  # Z-rotation with parameter
    return qml.expval(qml.PauliZ(0))  # Measure Z expectation value
```

**Key Components:**
- **Rotation Gates**: `RX`, `RY`, and `RZ` apply parameterized rotations around the X, Y, and Z axes
- **Parameter Scanning**: The code varies parameters to observe their impact on measurement outcomes
- **Bloch Vector Visualization**: Calculates and displays the Bloch sphere representation of the qubit state

**What we are getting:**
- How parameter changes affect quantum states
- Relationship between rotation parameters and measurement outcomes
- Visualizing quantum states on the Bloch sphere

## Project 2: Variational Quantum Eigensolver (VQE)

File: `variational_quantum_eigensolver.py`

### Description
This project implements VQE to find the ground state energy of a Heisenberg model Hamiltonian, demonstrating how variational circuits can solve eigenvalue problems.

### Code Breakdown

```python
# Ansatz definition (trial wavefunction)
def ansatz(params, wires):
    # Entangled state preparation
    qml.RY(params[0], wires=wires[0])  # Initial rotation on first qubit
    qml.RY(params[1], wires=wires[1])  # Initial rotation on second qubit
    qml.CNOT(wires=[wires[0], wires[1]])  # Entangle qubits
    
    # Additional rotations for expressivity
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])

# Hamiltonian definition
coeffs = [1.0, 1.0, 1.0]
observables = [
    qml.PauliX(0) @ qml.PauliX(1),  # X⊗X interaction
    qml.PauliY(0) @ qml.PauliY(1),  # Y⊗Y interaction
    qml.PauliZ(0) @ qml.PauliZ(1)   # Z⊗Z interaction
]
hamiltonian = qml.Hamiltonian(coeffs, observables)  # Heisenberg Hamiltonian

# Cost function (energy to minimize)
@qml.qnode(dev)
def cost_function(params):
    ansatz(params, wires=[0, 1])  # Prepare trial state
    return qml.expval(hamiltonian)  # Measure energy
```

**Key Components:**
- **Ansatz Design**: A parameterized circuit that prepares trial states
- **Hamiltonian Construction**: Defines the energy operator for the Heisenberg model
- **Optimization Loop**: Uses gradient descent to find parameters that minimize energy
- **Final State Analysis**: Examines the optimized quantum state and its energy

**What we are getting:**
- Creating and optimizing variational ansätze
- Encoding physical problems as Hamiltonians
- Gradient-based optimization of quantum circuits
- Quantum state preparation for approximating ground states

## Project 3: Quantum Neural Network

File: `quantum_neural_networks_binary_classification.py`

### Description
This project implements a quantum neural network for binary classification on the half-moons dataset, demonstrating how VQCs can be used for machine learning tasks.

### Code Breakdown

```python
# Quantum neural network definition
@qml.qnode(dev)
def circuit(inputs, weights):
    # Data encoding
    qml.RY(inputs[0], wires=0)  # Encode first feature into first qubit
    qml.RY(inputs[1], wires=1)  # Encode second feature into second qubit
    
    # Variational layers
    for l in range(n_layers):
        # Parameterized rotations
        for i in range(n_qubits):
            qml.RX(weights[l][i][0], wires=i)
            qml.RY(weights[l][i][1], wires=i)
        
        # Entanglement
        qml.CNOT(wires=[0, 1])  # Create quantum correlations
    
    # Measurement for classification
    return qml.expval(qml.PauliZ(0))

# Classification function
def classify(inputs, weights):
    return 1 if circuit(inputs, weights) > 0 else 0  # Binary decision

# Cost function
def cost(weights, X, y):
    predictions = []
    for x in X:
        predictions.append((circuit(x, weights) + 1) / 2)  # Scale from [-1,1] to [0,1]
    
    # Mean squared error calculation
    loss = 0
    for pred, target in zip(predictions, y):
        loss += (pred - target) ** 2
    
    return loss / len(y)
```

**Key Components:**
- **Data Encoding**: Converting classical data into quantum states
- **Parameterized Layers**: Multiple variational layers forming a "quantum neural network"
- **Entangling Operations**: Creating correlations between qubits
- **Training Loop**: Manual gradient descent implementation for optimization
- **Classification**: Converting quantum measurements to binary predictions

**What we are getting:**
- Encoding classical data in quantum states
- Building multi-layer quantum circuits
- Training quantum models for classification tasks
- Evaluating quantum model performance

## Requirements

To run these projects, you'll need:

```
pennylane
numpy
matplotlib
scikit-learn (for Project 3)
```

Install the requirements with:

```bash
pip install pennylane numpy matplotlib scikit-learn
```

## Path and Progression

These projects follow a logical progression:

1. **Project 1** introduces the building blocks of VQCs with simple parameterized gates
2. **Project 2** applies VQCs to physics problems and introduces optimization
3. **Project 3** extends VQCs to machine learning applications

By completing all three projects, you will gain a comprehensive understanding of variational quantum circuits and their applications in quantum computing.

# Variational Quantum Eigensolver (VQE) - Detailed Explanation

## What is a Variational Quantum Eigensolver?

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm designed to find the ground state energy of quantum systems. It combines a quantum computer's ability to efficiently represent and manipulate quantum states with a classical optimizer's ability to minimize a cost function. VQE has become one of the most promising near-term applications of quantum computing, particularly for quantum chemistry and materials science.

## Core Principles of VQE

### 1. Variational Principle

VQE is based on the variational principle in quantum mechanics, which states that:

For any trial wavefunction |ψ(θ)⟩ and a Hamiltonian H, the expectation value of the Hamiltonian is always greater than or equal to the ground state energy:

```
⟨ψ(θ)|H|ψ(θ)⟩ ≥ E₀
```

Where E₀ is the ground state energy. Equality holds if and only if |ψ(θ)⟩ is the ground state.

### 2. Hybrid Architecture

VQE leverages a hybrid quantum-classical approach:
- **Quantum Part**: Prepares and measures parameterized quantum states
- **Classical Part**: Optimizes the parameters to minimize energy

### 3. Parameterized Quantum Circuits

The core of VQE is a parameterized quantum circuit (also called an ansatz or trial wavefunction) that:
- Takes a set of parameters θ as input
- Prepares a quantum state |ψ(θ)⟩
- Can be systematically improved by adjusting parameters

## VQE Algorithm Steps

1. **Define the problem Hamiltonian H**
   - For chemistry: Typically derived from molecular structure using mappings like Jordan-Wigner or Bravyi-Kitaev
   - For physics: Constructed from model Hamiltonians (Heisenberg, Ising, etc.)

2. **Design a parameterized quantum circuit (ansatz)**
   - Hardware-efficient ansätze: Optimized for specific quantum hardware
   - Problem-inspired ansätze: Incorporate problem structure (e.g., UCCSD for chemistry)
   - Heuristic ansätze: Generic structures like layered circuits

3. **Measure the energy**
   - Decompose the Hamiltonian into measurable terms: H = Σᵢ cᵢ Pᵢ (where Pᵢ are Pauli strings)
   - For each term, measure its expectation value ⟨ψ(θ)|Pᵢ|ψ(θ)⟩
   - Compute the weighted sum: E(θ) = Σᵢ cᵢ ⟨ψ(θ)|Pᵢ|ψ(θ)⟩

4. **Optimize parameters**
   - Use a classical optimization algorithm to find parameters that minimize E(θ)
   - Common optimizers: COBYLA, SPSA, L-BFGS-B, gradient descent

5. **Analyze results**
   - Extract ground state energy and properties
   - Verify convergence and analyze error sources

## Mathematical Foundation

The mathematical objective of VQE is:

```
min₍θ₎ E(θ) = min₍θ₎ ⟨ψ(θ)|H|ψ(θ)⟩
```

Where:
- θ are the circuit parameters
- |ψ(θ)⟩ is the state prepared by the parameterized circuit
- H is the problem Hamiltonian
- E(θ) is the energy expectation value (cost function)

## Ansatz Design Considerations

The ansatz is crucial for VQE performance. An effective ansatz should:

1. **Be sufficiently expressive** - Able to represent or approximate the ground state
2. **Be efficiently implementable** - Use a reasonable number of gates
3. **Have a good parameter landscape** - Avoid barren plateaus and local minima
4. **Incorporate problem symmetries** - Respects conservation laws and symmetries
5. **Be hardware-aware** - Optimized for available quantum hardware connectivity and gate set

### Common Ansatz Types:

#### Hardware-Efficient Ansätze
```
Layer structure:
1. Single-qubit rotations on all qubits (RX, RY, RZ)
2. Entangling operations (CNOT, CZ, etc.)
3. Repeat layers for greater expressivity
```

#### Unitary Coupled Cluster (UCC)
Used primarily in quantum chemistry:
```
|ψ(θ)⟩ = e^(T(θ) - T†(θ)) |ϕ₀⟩

Where:
- |ϕ₀⟩ is an initial reference state (often Hartree-Fock)
- T(θ) contains excitation operators (single, double, etc.)
```

## Hamiltonian Decomposition

For a molecular Hamiltonian:

```
H = Σᵢⱼ hᵢⱼ aᵢ†aⱼ + Σᵢⱼₖₗ hᵢⱼₖₗ aᵢ†aⱼ†aₖaₗ + ...
```

We convert to qubit operators using mappings like Jordan-Wigner:

```
H = Σₘ cₘ Pₘ

Where:
- Pₘ are tensor products of Pauli operators (X, Y, Z, I)
- cₘ are real coefficients
```

## Optimization Strategies

### Gradient-Based Methods
- Analytical gradients: ∂E(θ)/∂θᵢ = ⟨ψ(θ)|∂H/∂θᵢ|ψ(θ)⟩
- Parameter-shift rule: Calculate gradients with finite differences
- Natural gradients: Account for the geometry of parameter space

### Gradient-Free Methods
- COBYLA (Constrained Optimization By Linear Approximation)
- Nelder-Mead simplex method
- SPSA (Simultaneous Perturbation Stochastic Approximation)

## Error Mitigation Techniques for VQE

1. **Zero-Noise Extrapolation (ZNE)**:
   - Run circuits at different noise levels
   - Extrapolate to zero-noise limit
   
2. **Symmetry Verification**:
   - Project measured states onto symmetry subspaces
   - Discard measurements that violate known symmetries

3. **Measurement Error Mitigation**:
   - Characterize and correct readout errors

## Implementing VQE with PennyLane

The basic structure of a VQE implementation with PennyLane looks like:

```python
import pennylane as qml
import numpy as np
from pennylane import qchem

# 1. Define quantum device
dev = qml.device('default.qubit', wires=4)

# 2. Define ansatz (trial wavefunction)
def ansatz(params, wires):
    # Initial state preparation
    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    
    # Variational layers
    for i in range(len(params)):
        qml.RY(params[i][0], wires=wires[0])
        qml.RY(params[i][1], wires=wires[1])
        qml.RY(params[i][2], wires=wires[2])
        qml.RY(params[i][3], wires=wires[3])
        
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CNOT(wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[2], wires[3]])

# 3. Define Hamiltonian
coeffs = [0.2, 0.5, 0.3]
observables = [
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliZ(2),
    qml.PauliY(2) @ qml.PauliZ(3)
]
hamiltonian = qml.Hamiltonian(coeffs, observables)

# 4. Define QNode for measuring energy
@qml.qnode(dev)
def cost_function(params):
    ansatz(params, wires=range(4))
    return qml.expval(hamiltonian)

# 5. Initialize parameters
np.random.seed(42)
num_layers = 2
init_params = np.random.uniform(0, 2*np.pi, (num_layers, 4))

# 6. Optimize parameters
max_iterations = 100
opt = qml.GradientDescentOptimizer(stepsize=0.1)

params = init_params
for i in range(max_iterations):
    params = opt.step(cost_function, params)
    if i % 10 == 0:
        print(f"Iteration {i}: Energy = {cost_function(params)}")

# 7. Final results
final_energy = cost_function(params)
print(f"Final energy: {final_energy}")
```

## Advanced VQE Techniques

### Adaptive Ansätze
- Grow the ansatz during optimization
- Add parameters that maximize gradient
- Prune parameters with low impact

### Subspace Expansion
- Expand around the VQE solution in a subspace
- Diagonalize Hamiltonian in this subspace
- Improves accuracy with minimal quantum resources

### Excited States VQE
- Witness-assisted VQE
- Orthogonality constraints
- Subspace search VQE

## Applications of VQE

### Quantum Chemistry
- Electronic structure problems
- Calculating molecular properties
- Drug discovery and materials design

### Condensed Matter Physics
- Finding ground states of spin systems
- Phase transitions in quantum materials
- Strongly correlated electron systems

### Optimization Problems
- Mapping combinatorial problems to Ising models
- Quantum approximate optimization algorithm (QAOA)

## Challenges and Limitations

1. **Measurement Requirements**
   - Need many measurements for precise energy estimation
   - Hamiltonian term growth with system size

2. **Optimization Landscapes**
   - Barren plateaus in random circuits
   - Local minima trapping

3. **Noise Sensitivity**
   - Circuit depth limitations on NISQ devices
   - Coherence time constraints

4. **Expressivity vs. Trainability**
   - More expressive ansätze often harder to train
   - Balance needed between representational power and trainability

## Future Directions

1. **Adaptive Error Mitigation**
   - Customized error mitigation strategies
   - Learning-based approaches

2. **Quantum-Classical Neural Networks**
   - Integration with classical neural networks
   - Quantum kernels and feature maps

3. **Hardware-Specific Optimizations**
   - Pulse-level control
   - Quantum optimal control theory

4. **Multi-Scale Approaches**
   - Classical pre-processing to reduce quantum resource requirements
   - Embedding methods for large systems

## Conclusion

The Variational Quantum Eigensolver represents one of the most promising approaches for achieving quantum advantage on near-term quantum computers. By combining the strengths of quantum and classical computation, VQE offers a practical path to solving important problems in quantum chemistry, materials science, and optimization that are intractable for classical computers alone.

Its hybrid nature makes it well-suited for NISQ (Noisy Intermediate-Scale Quantum) devices, and ongoing research continues to improve its efficiency, accuracy, and applicability to increasingly complex systems.