# Qontrast: Contrast Filters Mitigate Quantum Noise

## Abstract
We propose a novel method to mitigate measurement errors of quantum computations using image contrast filters, termed QONTRAST (Quantum Contrast filters). QONTRAST provides a general noise mitigation method at much lower cost andwithout requiring known expectation values than prior work. We experimentally evaluate QONTRAST for various benchmarks from prior work. Our results show fidelity improvements up to 18% for non-variational circuits like GHZ and Bernstein-Vazirani, and up to 5% for QAOA in SupermarQ benchmarks. Furthermore, hardware results in the state preparation of the Kagome lattice using VQE show a reduction of 22% in error rates. QONTRAST is platform agnostic across quantum device technology. Overall, results show that QONTRAST outperforms standard quantum resilience mechanisms of the Qiskit runtime, namely Twirled Readout Error eXtinction (TREX) and Zero Noise Extrapolation (ZNE).

