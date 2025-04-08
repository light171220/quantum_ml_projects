# Quantum Kernel Methods for Machine Learning

This repository contains three progressive projects demonstrating quantum kernel methods for classification tasks. These projects showcase how quantum computers can be leveraged to enhance machine learning through quantum feature spaces.

## Table of Contents

1. [Introduction to Quantum Kernels](#introduction-to-quantum-kernels)
2. [Project 1: Basic Quantum Kernel with Iris Dataset](#project-1-basic-quantum-kernel-with-iris-dataset)
3. [Project 2: ZZ Feature Map for Non-Linear Classification](#project-2-zz-feature-map-for-non-linear-classification)
4. [Project 3: Optimized Quantum Kernel with Kernel Alignment](#project-3-optimized-quantum-kernel-with-kernel-alignment)
5. [Mathematical Foundation](#mathematical-foundation)
6. [Implementation Details](#implementation-details)
7. [Experimental Results](#experimental-results)
8. [Extensions and Future Work](#extensions-and-future-work)
9. [Requirements and Setup](#requirements-and-setup)

## Introduction to Quantum Kernels

### What are Kernel Methods?

Kernel methods are a class of algorithms for pattern analysis that map data into a high-dimensional feature space, making linear separation easier. The "kernel trick" allows computing inner products in the feature space without explicitly computing the feature vectors, making them computationally efficient.

### Quantum Kernel Methods

Quantum kernel methods leverage quantum computers to compute kernels in exponentially large feature spaces that would be inaccessible to classical computers. The approach works by:

1. Encoding classical data into quantum states through a feature map circuit
2. Measuring the similarity between these quantum states (inner product)
3. Using this similarity measure as a kernel for classical machine learning algorithms

The advantage comes from quantum computers' ability to efficiently represent and manipulate high-dimensional data through superposition and entanglement.

## Project 1: Basic Quantum Kernel with Iris Dataset

### Overview

This introductory project implements a simple quantum kernel for binary classification on the Iris dataset. It demonstrates the basic workflow of quantum kernel methods.

### Key Components

#### Feature Map Circuit

```python
@qml.qnode(dev)
def feature_map(x):
    # Encode the features into a quantum state
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    
    # Add some entanglement
    qml.CNOT(wires=[0, 1])
    
    return qml.state()
```

This feature map encodes data using RY rotations and introduces entanglement with a CNOT gate.

#### Quantum Kernel Calculation

```python
def quantum_kernel(x1, x2):
    state1 = feature_map(x1)
    state2 = feature_map(x2)
    
    # Calculate |<ψ(x1)|ψ(x2)>|^2
    kernel_value = np.abs(np.vdot(state1, state2))**2
    return kernel_value
```

The kernel value is the squared absolute value of the inner product between quantum states, representing their similarity.

### Implementation Details

1. Load and preprocess the Iris dataset, selecting the first two features and two classes
2. Scale the features using StandardScaler
3. Define a quantum feature map that encodes data points into quantum states
4. Compute kernel matrices for training and testing data
5. Train an SVM classifier using the precomputed quantum kernel
6. Compare with a classical RBF kernel

### Results

The project demonstrates that even a simple quantum kernel can perform comparably to classical kernels on linearly separable data. It serves as a proof of concept for more complex quantum kernel implementations.

## Project 2: ZZ Feature Map for Non-Linear Classification

### Overview

This project implements a more sophisticated quantum kernel using the ZZ feature map, inspired by IBM's quantum kernel implementation, to handle non-linearly separable data (concentric circles).

### Key Components

#### ZZ Feature Map

```python
@qml.qnode(dev)
def zz_feature_map(x, reps=2):
    # First layer of Hadamards
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # Repeated blocks
    for r in range(reps):
        # Data-embedding
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        
        # ZZ entanglement + non-linear transformation
        qml.CNOT(wires=[0, 1])
        qml.RZ(x[0] * x[1], wires=1)  # Non-linear ZZ interaction
        qml.CNOT(wires=[0, 1])
        
        # Second round of single-qubit rotations
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
            qml.Hadamard(wires=i)
    
    return qml.state()
```

The ZZ feature map explicitly encodes non-linear interactions between features using the product term `x[0] * x[1]`, making it more expressive for non-linear data.

### Implementation Details

1. Generate a non-linearly separable dataset (concentric circles)
2. Scale the features to [0, 2π] range for quantum encoding
3. Implement the ZZ feature map that encodes both individual features and their products
4. Calculate kernel matrices and train an SVM classifier
5. Compare with a classical RBF kernel
6. Visualize the dataset and decision boundaries

### Results

The ZZ feature map demonstrates improved performance on non-linearly separable data compared to simpler feature maps. It effectively captures the circular decision boundary needed for the circles dataset.

## Project 3: Optimized Quantum Kernel with Kernel Alignment

### Overview

This advanced project implements a parameterized quantum kernel that can be optimized through kernel alignment. It demonstrates how to tune quantum circuits to improve classification performance on real-world data.

### Key Components

#### Parameterized Feature Map

```python
@qml.qnode(dev)
def parameterized_feature_map(x, params):
    # First layer - feature embedding
    for i in range(n_qubits):
        qml.RX(x[i], wires=i)
    
    # First entangling layer
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    
    # Parameterized rotation layer
    for i in range(n_qubits):
        qml.RZ(params[0, i], wires=i)
        qml.RY(params[1, i], wires=i)
    
    # Second entangling layer
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    
    # Second parameterized rotation layer
    for i in range(n_qubits):
        qml.RZ(params[2, i], wires=i)
        qml.RY(params[3, i], wires=i)
    
    return qml.state()
```

This feature map includes trainable parameters that can be optimized to improve performance.

#### Kernel Alignment

```python
def kernel_alignment(K, y):
    """Calculate alignment between kernel matrix K and target kernel from labels y"""
    # Construct target kernel: y_i * y_j (ideal kernel)
    n = len(y)
    y_matrix = np.outer(y, y)
    
    # Center the kernel matrix
    K_centered = K - np.mean(K, axis=0)
    
    # Calculate Frobenius inner product <K, y_matrix>
    alignment_num = np.sum(K_centered * y_matrix)
    
    # Calculate normalization terms
    K_norm = np.sqrt(np.sum(K_centered * K_centered))
    y_norm = np.sqrt(np.sum(y_matrix * y_matrix))
    
    # Return negative alignment (for minimization)
    alignment = alignment_num / (K_norm * y_norm)
    return -alignment  # Negative for minimization
```

Kernel alignment measures how well the quantum kernel matches an ideal kernel derived from the labels. Optimizing this alignment improves classification performance.

### Implementation Details

1. Load the Breast Cancer Wisconsin dataset and select significant features
2. Define a parameterized quantum feature map with tunable rotation angles
3. Implement kernel alignment as an optimization objective
4. Search for optimal parameters using a simple grid search strategy
5. Train an SVM using the optimized quantum kernel
6. Compare with a classical RBF kernel
7. Visualize the optimization process

### Results

The optimized quantum kernel demonstrates improved performance over both basic quantum kernels and classical kernels on the breast cancer dataset. This shows the potential for tuning quantum circuits to specific classification tasks.

## Mathematical Foundation

### Kernel Methods in Machine Learning

A kernel function $K(x_i, x_j)$ computes the inner product between data points $x_i$ and $x_j$ in some feature space $\phi$:

$$K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$$

In quantum computing, the feature map $\phi$ corresponds to a quantum circuit that encodes classical data into quantum states.

### Quantum State Fidelity as a Kernel

For quantum kernels, we define:

$$K_Q(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$$

Where $|\psi(x_i)\rangle$ represents the quantum state prepared by the feature map circuit for input $x_i$.

### Kernel Alignment

Kernel alignment measures how well a kernel matrix $K$ aligns with an ideal kernel matrix $K^*$ defined by the labels:

$$A(K, K^*) = \frac{\langle K, K^* \rangle_F}{\sqrt{\langle K, K \rangle_F \langle K^*, K^* \rangle_F}}$$

Where $\langle \cdot, \cdot \rangle_F$ denotes the Frobenius inner product. Higher alignment scores indicate better performance.

## Implementation Details

### Quantum Circuits Simulation

All quantum circuits are simulated using PennyLane, a Python library for quantum machine learning. The key simulation components include:

- QNodes: Quantum functions that run on quantum devices
- State vectors: Representations of quantum states
- Measurement operations: Converting quantum states to classical information

### Performance Considerations

Quantum kernel computation can be computationally intensive, especially for larger datasets. Performance optimizations include:

- Batched processing for large datasets
- Progress tracking for long-running computations
- Using subsets of data for optimization phases

## Experimental Results

Each project includes comparison with classical methods, demonstrating:

1. **Project 1**: Comparable performance to classical kernels on simple data
2. **Project 2**: Better handling of non-linear structure than simpler quantum kernels
3. **Project 3**: Improved performance through parameter optimization

## Extensions and Future Work

Potential extensions to these projects include:

1. **Advanced Feature Maps**: Implementing amplitude encoding or higher-order ZZZ interactions
2. **Improved Optimization**: Using gradient-based methods for kernel parameter optimization
3. **Noise Resilience**: Testing kernel performance under simulated quantum noise
4. **Hardware Implementation**: Running the kernels on actual quantum hardware
5. **Dimensionality Analysis**: Investigating how quantum kernels scale with data dimensionality

## Requirements and Setup

### Dependencies

```
pennylane>=0.23.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-kernel-methods.git
cd quantum-kernel-methods

# Install dependencies
pip install -r requirements.txt
```

### Running the Projects

```bash
# Project 1: Basic Quantum Kernel
python quantum_kernel_basic.py

# Project 2: ZZ Feature Map
python quantum_kernel_zz.py

# Project 3: Optimized Quantum Kernel
python quantum_kernel_optimized.py
```

## References

1. Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019).
2. Schuld, M., Killoran, N. Quantum machine learning in feature Hilbert spaces. Phys. Rev. Lett. 122, 040504 (2019).
3. Cristianini, N., Shawe-Taylor, J., Elisseeff, A., & Kandola, J. (2002). On kernel-target alignment. Advances in Neural Information Processing Systems, 14.
4. PennyLane documentation: https://pennylane.ai/qml/