# Quantum Fourier Transform: Mathematical Theory

## Introduction

The Quantum Fourier Transform (QFT) is the quantum analog of the classical Discrete Fourier Transform. It is a fundamental operation in quantum computing that transforms a quantum state in the computational basis into a state in the Fourier basis. The QFT plays a crucial role in many quantum algorithms, including Shor's factoring algorithm, quantum phase estimation, and various quantum machine learning algorithms.

## Mathematical Definition

### Formal Definition

The Quantum Fourier Transform on an $n$-qubit state $|j\rangle$ is defined as:

$$\text{QFT}_N |j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i jk/N} |k\rangle$$

where:
- $N = 2^n$ is the dimension of the Hilbert space
- $j$ is an integer in the range $0$ to $N-1$
- $|j\rangle$ and $|k\rangle$ are computational basis states

The QFT can also be expressed as a unitary operator:

$$\text{QFT}_N = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} \sum_{k=0}^{N-1} e^{2\pi i jk/N} |k\rangle \langle j|$$

### Relationship to Classical DFT

The classical DFT transforms a sequence of complex numbers $x_0, x_1, ..., x_{N-1}$ to another sequence $X_0, X_1, ..., X_{N-1}$ via:

$$X_k = \sum_{j=0}^{N-1} x_j e^{-2\pi i jk/N}$$

The QFT is analogous to this, but with key differences:
1. It transforms quantum states, not classical data
2. It preserves quantum superposition
3. The QFT uses $e^{+2\pi i jk/N}$ while the classical DFT uses $e^{-2\pi i jk/N}$

## Mathematical Properties

### Unitarity

The QFT is a unitary operation, meaning:

$$\text{QFT}^\dagger \text{QFT} = I$$

where $\text{QFT}^\dagger$ is the adjoint (or inverse) of QFT, and $I$ is the identity operator.

### Linearity

The QFT is a linear operation. For any linear combination of basis states:

$$\text{QFT}_N \left( \sum_j \alpha_j |j\rangle \right) = \sum_j \alpha_j \text{QFT}_N |j\rangle$$

### Inverse QFT

The inverse QFT is defined as:

$$\text{QFT}_N^{-1} |k\rangle = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} e^{-2\pi i jk/N} |j\rangle$$

## Factored Representation

A key insight for implementing the QFT efficiently is its factored representation. For an $n$-qubit state $|j\rangle$ where $j$ has the binary representation $j = j_1 j_2 \ldots j_n$ (with $j_1$ being the most significant bit), the QFT can be written as:

$$\text{QFT}_N |j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i jk/N} |k\rangle = \frac{1}{\sqrt{N}} \bigotimes_{l=1}^{n} \left( |0\rangle + e^{2\pi i 0.j_{n-l+1}...j_n} |1\rangle \right)$$

where $0.j_{n-l+1}...j_n$ is the binary fraction $0.j_{n-l+1}...j_n = j_{n-l+1}/2 + j_{n-l+2}/2^2 + ... + j_n/2^l$.

This factored form reveals that the QFT can be expressed as a tensor product of single-qubit states, each in a specific superposition determined by the binary representation of the input.

## Circuit Implementation

### Quantum Gates

The QFT is implemented using two types of quantum gates:

1. **Hadamard Gate ($H$)**: Transforms $|0\rangle$ to $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ and $|1\rangle$ to $\frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

2. **Controlled-Phase Gate ($R_k$)**: Applies a phase rotation

$$R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i/2^k} \end{pmatrix}$$

### Circuit Structure

For an $n$-qubit system, the QFT circuit consists of:

1. Apply a Hadamard gate to the first qubit
2. Apply controlled-$R_2$ gate from first qubit to second qubit
3. Apply controlled-$R_3$ gate from first qubit to third qubit
...and so on

The general structure for creating the QFT circuit is:

1. For each qubit $i$ from 1 to $n$:
   a. Apply Hadamard gate to qubit $i$
   b. For each qubit $j$ from $i+1$ to $n$:
      i. Apply controlled-$R_{j-i+1}$ from qubit $i$ (control) to qubit $j$ (target)

2. Finally, swap qubits to achieve the standard ordering: swap qubit 1 with qubit $n$, swap qubit 2 with qubit $n-1$, etc.

## Mathematical Analysis of the Circuit

Let's trace the state of a quantum system through the QFT circuit to show how it achieves the transformation.

Starting with an $n$-qubit basis state $|j\rangle$ where $j = j_1 j_2 \ldots j_n$ in binary:

1. After the Hadamard on the first qubit:
   $$|j_1 j_2 \ldots j_n\rangle \xrightarrow{H \text{ on qubit } 1} \frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i 0.j_1}|1\rangle) \otimes |j_2 \ldots j_n\rangle$$

2. After the controlled-phase gates from qubit 1:
   $$\frac{1}{\sqrt{2}}(|0\rangle + e^{2\pi i 0.j_1j_2\ldots j_n}|1\rangle) \otimes |j_2 \ldots j_n\rangle$$

3. Continuing for all qubits, we get:
   $$\frac{1}{\sqrt{2^n}}\bigotimes_{l=1}^{n} \left( |0\rangle + e^{2\pi i 0.j_{l}j_{l+1}\ldots j_n} |1\rangle \right)$$

4. After the swaps, we get:
   $$\frac{1}{\sqrt{2^n}}\bigotimes_{l=1}^{n} \left( |0\rangle + e^{2\pi i 0.j_{n-l+1}\ldots j_n} |1\rangle \right)$$

This is precisely the factored form of the QFT.

## Computational Complexity

The QFT requires:
- $n$ Hadamard gates
- $n(n-1)/2$ controlled-phase gates
- $\lfloor n/2 \rfloor$ SWAP gates

Therefore, the total number of gates is $O(n^2)$, which means the QFT can be performed with quadratic complexity in the number of qubits.

### Comparison with Classical FFT

The classical Fast Fourier Transform (FFT) requires $O(N \log N) = O(n 2^n)$ operations for an $N = 2^n$ dimensional input.

For an $n$-qubit system, the QFT requires only $O(n^2)$ gates, which is exponentially more efficient than the classical FFT. However, preparing the input state and measuring the output state must also be considered in the overall complexity analysis.

## Approximate QFT

For practical implementations, especially with large numbers of qubits, we can use the Approximate QFT (AQFT). The key insight is that controlled-phase gates with very small rotation angles (e.g., $R_k$ for large $k$) have minimal impact on the result.

The AQFT truncates the circuit by omitting phase rotations smaller than some threshold. This reduces the number of gates to $O(n \log n)$ with only a small loss in fidelity.

## Error Analysis

Sources of error in practical QFT implementations include:

1. **Gate Imperfections**: Physical implementations of quantum gates have inherent errors
2. **Decoherence**: Quantum states lose coherence over time
3. **Approximation Errors**: When using AQFT instead of exact QFT
4. **Measurement Errors**: When measuring the output state

The error in the QFT output can be quantified using the fidelity:

$$F(|\psi_{\text{ideal}}\rangle, |\psi_{\text{actual}}\rangle) = |\langle\psi_{\text{ideal}}|\psi_{\text{actual}}\rangle|^2$$

## Mathematical Applications

### Quantum Phase Estimation

In Quantum Phase Estimation, the QFT is used to extract the eigenvalue of a unitary operator. If $U|u\rangle = e^{2\pi i \phi}|u\rangle$, QPE can estimate $\phi$ with high precision.

The mathematical process is:
1. Prepare a superposition of counting qubits: $|+\rangle^{\otimes t} \otimes |u\rangle$
2. Apply controlled-$U^{2^j}$ operations
3. Apply the inverse QFT to the counting register
4. Measure the counting register to get an approximation of $\phi$

### Shor's Algorithm

In Shor's algorithm for factoring large integers, the QFT is used to find the period of a modular exponentiation function. 

For factoring an integer $N$, the algorithm:
1. Reduces the factoring problem to finding the period of $f(x) = a^x \bmod N$
2. Uses the QFT to efficiently determine this period
3. Uses the period to find factors of $N$ with high probability

## Advanced Mathematical Aspects

### QFT in Different Bases

While typically defined in the computational basis, the QFT can be generalized to other bases. For a general orthonormal basis $\{|b_j\rangle\}$, the QFT is:

$$\text{QFT}_N |b_j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i jk/N} |b_k\rangle$$

### Connection to Quantum Walks

The QFT has connections to quantum walks on certain graphs, particularly cyclic graphs. The eigenstates of the quantum walk operator are related to the Fourier basis states.

### Generalized QFT for Non-Binary Systems

The QFT can be generalized to quantum systems with dimensions that aren't powers of 2. For a system with dimension $D$, the generalized QFT is:

$$\text{QFT}_D |j\rangle = \frac{1}{\sqrt{D}} \sum_{k=0}^{D-1} e^{2\pi i jk/D} |k\rangle$$

### QFT in Continuous Variable Quantum Computing

In continuous variable quantum computing, the QFT corresponds to the optical Fourier transform, which can be implemented using optical components like beam splitters and phase shifters.

## Conclusion

The Quantum Fourier Transform is a fundamental operation in quantum computing with profound mathematical elegance. Its ability to efficiently transform quantum states makes it a key component of many quantum algorithms that demonstrate exponential speedup over classical algorithms. The mathematical theory of the QFT bridges quantum mechanics, computational complexity, and Fourier analysis, providing insights into the power and limitations of quantum computation.