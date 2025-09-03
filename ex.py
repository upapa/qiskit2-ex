# from qiskit.circuit import QuantumCircuit
# from qiskit.transpiler import generate_preset_pass_manager
# from qiskit_ibm_runtime import SamplerV2 as Sampler
# from qiskit_ibm_runtime.fake_provider import FakeManilaV2
 
# # Bell Circuit
# qc = QuantumCircuit(2)
# qc.h(0)
# qc.cx(0, 1)
# qc.measure_all()
 
# # Run the sampler job locally using FakeManilaV2
# fake_manila = FakeManilaV2()
# pm = generate_preset_pass_manager(backend=fake_manila, optimization_level=1)
# isa_qc = pm.run(qc)
 
# # You can use a fixed seed to get fixed results.
# options = {"simulator": {"seed_simulator": 42}}
# sampler = Sampler(mode=fake_manila, options=options)
 
# result = sampler.run([isa_qc]).result()
# print(result[0])

from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
 
# Bell Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
 
# Run the sampler job locally using AerSimulator.
# Session syntax is supported but ignored because local mode doesn't support sessions.
aer_sim = AerSimulator()
pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
isa_qc = pm.run(qc)
with Session(backend=aer_sim) as session:
    sampler = Sampler(mode=session)
    result = sampler.run([isa_qc]).result()
print(result[0])