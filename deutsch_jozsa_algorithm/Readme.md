# Deutsch-Jozsa Algorithm

## Overview

The Deutsch-Jozsa algorithm, formulated by David Deutsch and Richard Jozsa in 1992, is one of the first quantum algorithms to demonstrate a provable exponential advantage over classical algorithms for a specific problem.

This repository contains a PennyLane implementation of the Deutsch-Jozsa algorithm, which determines whether a black-box function (oracle) is constant or balanced with a single query.

## Problem Statement

Given a black-box function $f: \{0,1\}^n \rightarrow \{0,1\}$ that is promised to be either:
- **Constant**: Returns the same output for all inputs (either always 0 or always 1)
- **Balanced**: Returns 0 for exactly half of all possible inputs and 1 for the other half

The task is to determine whether $f$ is constant or balanced.

## Classical vs. Quantum Complexity

- **Classical**: Requires $2^{n-1} + 1$ evaluations of $f$ in the worst case
- **Quantum**: Requires only 1 evaluation of $f$ regardless of $n$

## Mathematical Foundation

### Quantum State Representation

For an $n$-qubit system, a general state can be expressed as:

$$|\psi\rangle = \sum_{x \in \{0,1\}^n} \alpha_x |x\rangle$$

where $\alpha_x$ are complex amplitudes with $\sum_{x} |\alpha_x|^2 = 1$.

### Hadamard Transformation

The Hadamard gate transforms a single qubit as follows:
- $H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$
- $H|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$

When applied to $n$ qubits initially in the $|0\rangle$ state, the result is a superposition of all possible $n$-bit strings:

$$H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}}\sum_{x \in \{0,1\}^n} |x\rangle$$

### Oracle Implementation

The oracle function $f(x)$ is implemented as a unitary transformation $U_f$ that acts on an $(n+1)$-qubit system as:

$$U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$$

where $\oplus$ represents addition modulo 2.

### Algorithm Steps and Mathematical Analysis

1. **Initial State Preparation**:
   Start with $n+1$ qubits in the state $|0\rangle^{\otimes n}|1\rangle$.

2. **Apply Hadamard Gates**:
   Apply Hadamard gates to all qubits to get:
   $$H^{\otimes n}|0\rangle^{\otimes n} \otimes H|1\rangle = \frac{1}{\sqrt{2^n}}\sum_{x \in \{0,1\}^n} |x\rangle \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

3. **Apply Oracle Function**:
   Apply $U_f$ to get:
   $$U_f\left(\frac{1}{\sqrt{2^n}}\sum_{x} |x\rangle \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)\right) = \frac{1}{\sqrt{2^{n+1}}}\sum_{x} (-1)^{f(x)}|x\rangle \otimes (|0\rangle - |1\rangle)$$

   This simplifies to:
   $$\frac{1}{\sqrt{2^n}}\sum_{x} (-1)^{f(x)}|x\rangle \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

4. **Apply Hadamard Gates to First $n$ Qubits**:
   $$H^{\otimes n}\left(\frac{1}{\sqrt{2^n}}\sum_{x} (-1)^{f(x)}|x\rangle\right) \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

   This gives:
   $$\frac{1}{2^n}\sum_{y}\sum_{x} (-1)^{f(x) + x \cdot y}|y\rangle \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$
   
   where $x \cdot y$ represents the bitwise dot product of $x$ and $y$.

5. **Measurement**:
   Measure the first $n$ qubits.

### Outcome Analysis

- If $f$ is **constant**, the amplitude of $|0\rangle^{\otimes n}$ is $\pm 1$, so measuring all zeros occurs with probability 1.
- If $f$ is **balanced**, the amplitude of $|0\rangle^{\otimes n}$ is 0, so measuring all zeros never occurs.

Therefore, if the measurement gives all zeros, $f$ is constant; otherwise, $f$ is balanced.

## Implementation Details

The code implements the Deutsch-Jozsa algorithm using PennyLane, a quantum machine learning library. It includes:

- Oracle implementations for constant and balanced functions
- Quantum circuit construction
- Measurement and interpretation of results

## Usage

```python
import pennylane as qml
import numpy as np

# Run the algorithm
print("Testing Deutsch-Jozsa algorithm with different oracles:")
print(f"Constant oracle (always 0): {'Constant' if is_constant(constant_0_oracle) else 'Balanced'}")
print(f"Constant oracle (always 1): {'Constant' if is_constant(constant_1_oracle) else 'Balanced'}")
print(f"Balanced oracle (example 1): {'Constant' if is_constant(balanced_oracle_1) else 'Balanced'}")
print(f"Balanced oracle (example 2): {'Constant' if is_constant(balanced_oracle_2) else 'Balanced'}")
```

## Requirements

- PennyLane
- NumPy

## References

1. Deutsch, D., & Jozsa, R. (1992). Rapid solution of problems by quantum computation. Proceedings of the Royal Society of London. Series A: Mathematical and Physical Sciences, 439(1907), 553-558.
2. Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information: 10th Anniversary Edition. Cambridge University Press.