from qiskit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2 as Estimator
import numpy as np

def qft_circuit(n):
    """n-qubit Quantum Fourier Transform circuit"""
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j+1, n):
            angle = np.pi / 2**(k-j)
            qc.cp(angle, k, j)
    # Swap qubits to reverse order
    for i in range(n//2):
        qc.swap(i, n-i-1)
    return qc

if __name__ == "__main__":
    n_qubits = 3
    qc = qft_circuit(n_qubits)
    qc.draw("mpl").show()
    # # Simulate the circuit
    # backend = Aer.get_backend('statevector_simulator')
    # job = execute(qc, backend)
    # result = job.result()
    # statevector = result.get_statevector()
    # print("Statevector:", statevector)