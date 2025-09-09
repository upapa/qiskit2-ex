p = 0.2

import numpy as np
from qiskit import QuantumCircuit 

class BernoulliA(QuantumCircuit):
    def __init__(self, p):
        super().__init__(1)
        theta_p = 2 * np.arcsin(np.sqrt(p))
        self.ry(theta_p, 0)
        
class BernoulliQ(QuantumCircuit):
    def __init__(self, p):
        super().__init__(1)
        self._theta_p = 2 * np.arcsin(np.sqrt(p))
        self.ry(2*self._theta_p, 0)
    def power(self, k):
        q_k = QuantumCircuit(1)
        q_k.ry(2*k*self._theta_p, 0)
        return q_k
    
A = BernoulliA(p)
Q = BernoulliQ(p)

from qiskit_algorithms import EstimationProblem 

problem = EstimationProblem(
    state_preparation=A,
    grover_operator=Q,
    objective_qubits=[0],
)

from qiskit.primitives import StatevectorSampler as Sampler
sampler = Sampler()

from qiskit_algorithms import AmplitudeEstimation
ae = AmplitudeEstimation(
    num_eval_qubits=3,
    sampler=sampler
)
ae_result = ae.estimate(problem)
print(ae_result.estimation)
print("Interpolated MLE estimator:", ae_result.mle)


from qiskit_algorithms import IterativeAmplitudeEstimation

iae = IterativeAmplitudeEstimation(
    epsilon_target=0.01,  # target accuracy
    alpha=0.05,  # width of the confidence interval
    sampler=sampler,
)
iae_result = iae.estimate(problem)

print("Estimate:", iae_result.estimation)


from qiskit_algorithms import MaximumLikelihoodAmplitudeEstimation

mlae = MaximumLikelihoodAmplitudeEstimation(
    evaluation_schedule=3,  # log2 of the maximal Grover power
    sampler=sampler,
)
mlae_result = mlae.estimate(problem)

print("Estimate:", mlae_result.estimation)

from qiskit_algorithms import FasterAmplitudeEstimation

fae = FasterAmplitudeEstimation(
    delta=0.01,  # target accuracy
    maxiter=3,  # determines the maximal power of the Grover operator
    sampler=sampler,
)
fae_result = fae.estimate(problem)

print("Estimate:", fae_result.estimation)