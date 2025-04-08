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