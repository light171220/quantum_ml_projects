# Quantum Phase Estimation: Mathematical Theory

## Introduction

Quantum Phase Estimation (QPE) is a fundamental quantum algorithm that determines the eigenvalue of a unitary operator. More precisely, given a unitary operator $U$ and an eigenstate $|u\rangle$ such that $U|u\rangle = e^{2\pi i\phi}|u\rangle$, QPE estimates the phase $\phi \in [0,1)$ with high precision. This algorithm is a cornerstone of quantum computing and serves as a subroutine in many important quantum algorithms, including Shor's factoring algorithm, quantum counting, and quantum simulation.

## Mathematical Framework

### Eigenvalue Problem

In quantum mechanics, the eigenvalue equation for a unitary operator $U$ can be written as:

$$U|u\rangle = \lambda|u\rangle$$

where $|u\rangle$ is an eigenstate and $\lambda$ is the corresponding eigenvalue. Since $U$ is unitary, $|\lambda| = 1$, so we can write:

$$\lambda = e^{2\pi i\phi}$$

where $\phi \in [0,1)$ is the phase we want to estimate.

### Quantum Circuit Structure

The QPE algorithm uses two quantum registers:
1. The **counting register** with $t$ qubits (used to store the estimate of $\phi$)
2. The **eigenstate register** with enough qubits to store $|u\rangle$

The algorithm is structured as follows:

1. Initialize the counting register to $|0\rangle^{\otimes t}$ and the eigenstate register to $|u\rangle$
2. Apply Hadamard gates to all qubits in the counting register
3. Apply controlled-$U^{2^j}$ operations
4. Apply the inverse Quantum Fourier Transform to the counting register
5. Measure the counting register to obtain an estimate of $\phi$

## Mathematical Derivation

### Initial State

The initial state of the system is:

$$|\psi_0\rangle = |0\rangle^{\otimes t} \otimes |u\rangle$$

### After Hadamard Gates

After applying Hadamard gates to the counting register:

$$|\psi_1\rangle = \frac{1}{\sqrt{2^t}} \sum_{j=0}^{2^t-1} |j\rangle \otimes |u\rangle$$

### Controlled-$U$ Operations

We then apply controlled-$U^{2^j}$ operations, where the $j$-th qubit in the counting register controls the application of $U^{2^j}$ to the eigenstate register.

For a basis state $|j\rangle$ in the counting register, where $j$ has binary representation $j = j_0 + j_1 \cdot 2 + j_2 \cdot 2^2 + \ldots + j_{t-1} \cdot 2^{t-1}$, the effect is:

$$|j\rangle \otimes |u\rangle \rightarrow |j\rangle \otimes U^{j_0 \cdot 2^0} U^{j_1 \cdot 2^1} \ldots U^{j_{t-1} \cdot 2^{t-1}} |u\rangle$$

Since $U|u\rangle = e^{2\pi i\phi}|u\rangle$, we have $U^{2^k}|u\rangle = e^{2\pi i\phi \cdot 2^k}|u\rangle$. Therefore:

$$|j\rangle \otimes |u\rangle \rightarrow |j\rangle \otimes e^{2\pi i\phi \cdot j}|u\rangle = e^{2\pi i\phi \cdot j}|j\rangle \otimes |u\rangle$$

For the entire superposition:

$$|\psi_2\rangle = \frac{1}{\sqrt{2^t}} \sum_{j=0}^{2^t-1} e^{2\pi i\phi \cdot j}|j\rangle \otimes |u\rangle$$

### Inverse Quantum Fourier Transform

The inverse QFT transforms $|j\rangle$ as follows:

$$\text{QFT}^{-1}|j\rangle = \frac{1}{\sqrt{2^t}} \sum_{k=0}^{2^t-1} e^{-2\pi i jk/2^t}|k\rangle$$

Applying this to our state $|\psi_2\rangle$:

$$\begin{align*}
|\psi_3\rangle &= \frac{1}{\sqrt{2^t}} \sum_{j=0}^{2^t-1} e^{2\pi i\phi \cdot j} \left( \frac{1}{\sqrt{2^t}} \sum_{k=0}^{2^t-1} e^{-2\pi i jk/2^t}|k\rangle \right) \otimes |u\rangle \\
&= \frac{1}{2^t} \sum_{k=0}^{2^t-1} \left( \sum_{j=0}^{2^t-1} e^{2\pi i j(\phi - k/2^t)} \right) |k\rangle \otimes |u\rangle
\end{align*}$$

The inner sum is a geometric series. If $\phi = k/2^t$ exactly, this sum equals $2^t$. Otherwise, it concentrates around values of $k$ that make $\phi \approx k/2^t$.

### Measurement Outcome Analysis

After measurement of the counting register, we obtain some value $k$ with probability:

$$P(k) = \left| \frac{1}{2^t} \sum_{j=0}^{2^t-1} e^{2\pi i j(\phi - k/2^t)} \right|^2$$

This probability is highest when $k/2^t$ is closest to $\phi$.

## Precision and Success Probability

### Binary Fraction Representation

Let's represent $\phi$ in its binary fraction:

$$\phi = 0.b_1b_2\ldots b_m\ldots$$

where $b_i \in \{0,1\}$ are binary digits. With $t$ qubits in the counting register, we can estimate $\phi$ up to $t$ bits of precision.

### Error Bound

If $\phi$ can be expressed exactly with $s$ bits where $s \leq t$, i.e., $\phi = 0.b_1b_2\ldots b_s$, then QPE will give the exact value with certainty.

If $\phi$ requires more than $t$ bits, QPE will find the best $t$-bit approximation with high probability. Specifically, QPE will output a $t$-bit approximation $\tilde{\phi}$ such that $|\phi - \tilde{\phi}| \leq 2^{-t}$ with probability at least $1 - \epsilon$ if we use $t + \log(2 + 1/(2\epsilon))$ qubits in the counting register.

### Probability Distribution

The probability of measuring a particular value $k$ in the counting register is:

$$P(k) = \frac{1}{2^{2t}} \left| \frac{\sin(\pi 2^t(\phi - k/2^t))}{\sin(\pi(\phi - k/2^t))} \right|^2$$

This distribution is sharply peaked around the values of $k$ that make $k/2^t$ close to $\phi$.

## Mathematical Extensions and Optimizations

### Phase Kickback Mechanism

The controlled-$U$ operations in QPE work through a quantum phenomenon called "phase kickback". For an eigenstate $|u\rangle$ with $U|u\rangle = e^{2\pi i\phi}|u\rangle$, we have:

$$\text{CTRL-}U|c\rangle|u\rangle = |c\rangle U^c |u\rangle = |c\rangle e^{2\pi i\phi \cdot c}|u\rangle = e^{2\pi i\phi \cdot c}|c\rangle|u\rangle$$

This means the phase gets "kicked back" from the target register to the control qubit.

### Iterative Phase Estimation

For some applications, we don't need all bits of $\phi$ at once. Iterative Phase Estimation (IPE) determines $\phi$ one bit at a time, starting with the most significant bit.

For the $j$th bit, the circuit is:
- Apply Hadamard to a single control qubit
- Apply controlled-$U^{2^{t-j}}$ 
- Apply a rotation $R_z(-\pi/2^{j-1})$ if previous bits suggest it
- Apply Hadamard again
- Measure to determine the $j$th bit of $\phi$

### Inexact Eigenstates

If the input state is not an exact eigenstate but a superposition $|\psi\rangle = \sum_i \alpha_i |u_i\rangle$ of eigenstates with $U|u_i\rangle = e^{2\pi i\phi_i}|u_i\rangle$, QPE will return each phase $\phi_i$ with probability $|\alpha_i|^2$.

## Error Analysis

### Sources of Error

1. **Discretization Error**: With $t$ qubits, we can only represent phases with precision $2^{-t}$
2. **Algorithmic Error**: Even with perfect quantum operations, there's an intrinsic probability of error in the algorithm
3. **Implementation Error**: Real quantum devices have gate errors, decoherence, and measurement errors

### Error Mitigation Strategies

1. **Increased Register Size**: Using more qubits in the counting register reduces discretization error
2. **Repeated Measurements**: Running QPE multiple times and taking the majority vote can reduce algorithmic error
3. **Error Correction**: Quantum error correction can mitigate implementation errors

## Mathematical Applications

### Eigenvalue Estimation

QPE directly provides the eigenvalues of a unitary operator, which is useful for quantum simulation of physical systems.

### Period Finding

QPE can find the period of a periodic function $f(x) = f(x+r)$. This is used in Shor's algorithm, where the period of a modular exponentiation function reveals factors of large numbers.

### Order Finding

For a function $f(x) = a^x \mod N$, QPE can find the order of $a$ modulo $N$ (the smallest positive integer $r$ such that $a^r \equiv 1 \mod N$).

### Quantum Counting

QPE can be used to count the number of solutions to a search problem by estimating the eigenvalues of the Grover diffusion operator.

## Quantum Circuit Implementation

### Basic Circuit

For a unitary $U$ with eigenvalue $e^{2\pi i\phi}$ and eigenstate $|u\rangle$:

1. **Register Initialization**:
   - Initialize counting register: $|0\rangle^{\otimes t}$
   - Initialize eigenstate register: $|u\rangle$

2. **Hadamard Gates**:
   - Apply $H^{\otimes t}$ to counting register: $\frac{1}{\sqrt{2^t}} \sum_{j=0}^{2^t-1} |j\rangle \otimes |u\rangle$

3. **Controlled-U Operations**:
   - Apply controlled-$U^{2^j}$ operations: $\frac{1}{\sqrt{2^t}} \sum_{j=0}^{2^t-1} e^{2\pi i\phi \cdot j}|j\rangle \otimes |u\rangle$

4. **Inverse QFT**:
   - Apply inverse QFT to counting register
   
5. **Measurement**:
   - Measure counting register to get estimate of $\phi$

### Matrix Representation

For a 1-qubit counting register and assuming $U$ is a single-qubit gate, the circuit matrix representation is:

$\text{QPE} = (QFT^{-1} \otimes I) \cdot CU \cdot (H \otimes I) \cdot (|0\rangle \otimes |u\rangle)$

where $CU$ represents the controlled-$U$ operation.

## Complexity Analysis

### Computational Complexity

The complexity of QPE depends on:
1. The number of qubits in the counting register: $t$
2. The complexity of implementing controlled-$U^{2^j}$ operations

For a unitary $U$ that can be implemented using $O(n)$ basic gates:
- The controlled-$U^{2^j}$ operations require $O(t \cdot n \cdot 2^t)$ gates in the worst case
- The QFT requires $O(t^2)$ gates
- The total gate complexity is $O(t \cdot n \cdot 2^t + t^2)$

### Space Complexity

QPE requires:
- $t$ qubits for the counting register
- Enough qubits to represent the eigenstate $|u\rangle$

## Advanced Topics and Extensions

### Bayesian Phase Estimation

Bayesian Phase Estimation (BPE) uses Bayesian inference to update our knowledge of the phase through sequential measurements, often requiring fewer qubits than standard QPE.

### Amplitude Estimation

Quantum Amplitude Estimation (QAE) is a generalization of QPE that estimates the amplitude of a target state in a quantum superposition. For an operator $A$ and initial state $|0\rangle$ with $A|0\rangle = \sin(\theta)|1\rangle + \cos(\theta)|0'\rangle$, QAE estimates $\theta$.

### Phase Estimation with Coherent States

In continuous-variable quantum computing, phase estimation can be performed using coherent states and displacement operators.

## Mathematical Equivalence to Classical Methods

The QPE algorithm has connections to classical algorithms:

1. **Classical Fourier Analysis**: QPE uses the quantum Fourier transform to extract frequency information, similar to classical Fourier analysis

2. **Maximum Likelihood Estimation**: The measurement outcome in QPE is effectively a maximum likelihood estimate of the phase

3. **Monte Carlo Methods**: The probabilistic nature of QPE has similarities to Monte Carlo methods in classical computing

## Theoretical Bounds

### Lower Bounds

The quantum phase estimation algorithm achieves the Heisenberg limit for phase estimation, which is the theoretical lower bound for the uncertainty in estimating a phase with $N$ quantum resources:

$\Delta \phi \sim \frac{1}{N}$

In contrast, classical methods are limited by the standard quantum limit:

$\Delta \phi \sim \frac{1}{\sqrt{N}}$

### Optimality Results

For estimating an arbitrary phase $\phi$ with precision $\epsilon$ and success probability at least $1-\delta$, any quantum algorithm requires at least $\Omega(\log(1/\delta)/\epsilon)$ applications of the controlled-$U$ operation.

QPE achieves $O(\log(1/\delta)/\epsilon)$ applications, making it asymptotically optimal.

## Conclusion

Quantum Phase Estimation is a powerful quantum algorithm that leverages quantum parallelism and the quantum Fourier transform to efficiently extract phase information from unitary operators. Its ability to determine eigenvalues with exponential precision forms the mathematical foundation for many of quantum computing's most promising applications, from factoring large numbers to simulating quantum systems.

The algorithm's mathematical elegance demonstrates how quantum computing can exploit the principles of quantum mechanics to achieve computational advantages over classical methods, particularly for problems with an underlying group-theoretic structure.