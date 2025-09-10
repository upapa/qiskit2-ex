# quantum phase estimation 
# ref: https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/quantum-phase-estimation.ipynb

#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
# from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools and circuits
from qiskit.visualization import plot_histogram, plot_distribution
from qiskit.circuit.library import QFT, QFTGate

num_qubits = 15
qpe = QuantumCircuit(num_qubits+1, num_qubits)  # 1arg: num of qubits, 2arg: num of classical bits(measuring) 
qpe.x(num_qubits)
# print(qpe.draw())

for qubit in range(num_qubits):
    qpe.h(qubit)
# print(qpe.draw())

angle = 2*math.pi/3 
repetitions = 1
for counting_qubit in range(num_qubits):
    for i in range(repetitions):
        qpe.cp(angle, counting_qubit, num_qubits); # controlled-T
    repetitions *= 2
# print(qpe.draw())

qpe.barrier()
# Apply inverse QFT
# qpe = qpe.compose(QFT(3, inverse=True), [0,1,2])

from qiskit.synthesis.qft import synth_qft_full
qpe = qpe.compose(synth_qft_full(num_qubits, inverse=True), range(num_qubits))
# Measure
qpe.barrier()
for n in range(num_qubits):
    qpe.measure(n,n)

# print(qpe.draw())

# # back end: version2에서 많이 수정된 부분
# aer_sim = Aer.get_backend('aer_simulator')
# shots = 2048
# t_qpe = transpile(qpe, aer_sim)
# results = aer_sim.run(t_qpe, shots=shots).result()
# answer = results.get_counts()

# version 2.1.2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
# fake backend는 종류에 따라서 qubit 최대 수의 제한이 있음
# ref: https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeFez, FakeGuadalupeV2, FakeSherbrooke
backend = FakeGuadalupeV2()  # 16 qubits
# backend = FakeManilaV2  # 5 qubits
# backend = FakeFez  # 156 qubits
# backend = FakeSherbrooke # 126 qubits

# # Convert to an ISA circuit and layout-mapped observables.
# pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
# isa_circuit = pm.run(qpe)
# mapped_observables = []
# # estimator = Estimator(backend)
# # job = estimator.run([(isa_circuit, mapped_observables)])
# # answer = job.result()[0]

# from qiskit import transpile
options = {"simulator": {"seed_simulator": 42}}
# transpiled_circuit = transpile(qpe, backend)
# job = sampler.run([transpiled_circuit])
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
isa_qpe = pm.run(qpe)

sampler = Sampler(mode=backend)
job = sampler.run([isa_qpe], shots=4096) 
pub_result = job.result()[0]


# from qiskit_aer import AerSimulator
# aer_sim = AerSimulator()
# sampler = Sampler(mode=aer_sim)
# job = sampler.run([isa_qpe])
# pub_result = job.result()

# print(pub_result)
answer = pub_result.data.c.get_counts()
print(answer)
plot_distribution(answer).show()