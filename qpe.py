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
from qiskit.circuit.library import QFT

qpe = QuantumCircuit(4, 3)
qpe.x(3)
# print(qpe.draw())

for qubit in range(3):
    qpe.h(qubit)
# print(qpe.draw())

repetitions = 1
for counting_qubit in range(3):
    for i in range(repetitions):
        qpe.cp(math.pi/4, counting_qubit, 3); # controlled-T
    repetitions *= 2
# print(qpe.draw())

qpe.barrier()
# Apply inverse QFT
qpe = qpe.compose(QFT(3, inverse=True), [0,1,2])
# Measure
qpe.barrier()
for n in range(3):
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
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
backend = FakeManilaV2()


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
# sampler = Sampler(mode=backend,options=options)
# job = sampler.run([isa_qpe]) 
# pub_result = job.result()[0]
# answer = pub_result.data.meas.get_counts()  # data에 meas 필드가 없다. 실행에 실패한걸까?
from qiskit_aer import AerSimulator
aer_sim = AerSimulator()
sampler = Sampler(mode=aer_sim)
job = sampler.run([isa_qpe])
pub_result = job.result()
print(pub_result)
# answer = pub_result.data.meas.get_counts()
# print(answer)
# plot_distribution(answer)