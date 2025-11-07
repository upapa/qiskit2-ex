```python
import qiskit
print(qiskit.__version__)
import qiskit, qiskit_aer, qiskit_algorithms
print("qiskit:", qiskit.__version__)
print("qiskit-aer:", qiskit_aer.__version__)
print("qiskit-algorithms:", qiskit_algorithms.__version__)
```

    2.2.3
    qiskit: 2.2.3
    qiskit-aer: 0.17.2
    qiskit-algorithms: 0.4.0



```python
import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np

from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunctionGate
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from qiskit_aer.primitives import SamplerV2
from qiskit.primitives import StatevectorSampler
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit import ClassicalRegister
```


```python
# number of qubits to represent the uncertainty
num_uncertainty_qubits = 3

# parameters for considered random distribution
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.05  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity

# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
)
```


```python
# plot probability distribution
x = uncertainty_model.values
y = uncertainty_model.probabilities
plt.bar(x, y, width=0.2)
plt.xticks(x, size=15, rotation=90)
plt.yticks(size=15)
plt.grid()
plt.xlabel("Spot Price at Maturity $S_T$ (\$)", size=15)
plt.ylabel("Probability ($\%$)", size=15)
plt.show()
```

    <>:8: SyntaxWarning: invalid escape sequence '\$'
    <>:9: SyntaxWarning: invalid escape sequence '\%'
    <>:8: SyntaxWarning: invalid escape sequence '\$'
    <>:9: SyntaxWarning: invalid escape sequence '\%'
    /var/folders/nt/1tswn65j7qv_t7dq9q7cctjw0000gn/T/ipykernel_5564/2885665035.py:8: SyntaxWarning: invalid escape sequence '\$'
      plt.xlabel("Spot Price at Maturity $S_T$ (\$)", size=15)
    /var/folders/nt/1tswn65j7qv_t7dq9q7cctjw0000gn/T/ipykernel_5564/2885665035.py:9: SyntaxWarning: invalid escape sequence '\%'
      plt.ylabel("Probability ($\%$)", size=15)



    
![png](output_3_1.png)
    



```python
# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 1.896

# set the approximation scaling for the payoff function
c_approx = 0.25

# setup piecewise linear objective fcuntion
breakpoints = [low, strike_price]
slopes = [0, 1]
offsets = [0, 0]
f_min = 0
f_max = high - strike_price
european_call_objective = LinearAmplitudeFunctionGate(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low, high),
    image=(f_min, f_max),
    breakpoints=breakpoints,
    rescaling_factor=c_approx,
)

# european_call_objective = LinearAmplitudeFunction(
#     num_uncertainty_qubits,
#     slopes,
#     offsets,
#     domain=(low, high),
#     image=(f_min, f_max),
#     breakpoints=breakpoints,
#     rescaling_factor=c_approx,
# )

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
num_qubits = european_call_objective.num_qubits
european_call = QuantumCircuit(num_qubits)
european_call.append(uncertainty_model, range(num_uncertainty_qubits))
european_call.append(european_call_objective, range(num_qubits))

# draw the circuit
european_call.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     ┌───────┐┌────┐
q_0: ┤0      ├┤0   ├
     │       ││    │
q_1: ┤1 P(X) ├┤1   ├
     │       ││    │
q_2: ┤2      ├┤2 F ├
     └───────┘│    │
q_3: ─────────┤3   ├
              │    │
q_4: ─────────┤4   ├
              └────┘</pre>




```python
# plot exact payoff function (evaluated on the grid of the uncertainty model)
x = uncertainty_model.values
y = np.maximum(0, x - strike_price)
plt.plot(x, y, "ro-")
plt.grid()
plt.title("Payoff Function", size=15)
plt.xlabel("Spot Price", size=15)
plt.ylabel("Payoff", size=15)
plt.xticks(x, size=15, rotation=90)
plt.yticks(size=15)
plt.show()
```


    
![png](output_5_0.png)
    



```python

# evaluate exact expected value (normalized to the [0, 1] interval)
exact_value = np.dot(uncertainty_model.probabilities, y)
exact_delta = sum(uncertainty_model.probabilities[x >= strike_price])
print("exact expected value:\t%.4f" % exact_value)
print("exact delta value:   \t%.4f" % exact_delta)
```

    exact expected value:	0.1623
    exact delta value:   	0.8098



```python
european_call.draw()

```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     ┌───────┐┌────┐
q_0: ┤0      ├┤0   ├
     │       ││    │
q_1: ┤1 P(X) ├┤1   ├
     │       ││    │
q_2: ┤2      ├┤2 F ├
     └───────┘│    │
q_3: ─────────┤3   ├
              │    │
q_4: ─────────┤4   ├
              └────┘</pre>




```python
# set target precision and confidence level
epsilon = 0.01
alpha = 0.05


# sampler = Sampler(
#     backend_options={"method": "statevector"},
#     run_options={"shots": 1000, "seed_simulator": 75}
# )

sampler = StatevectorSampler()


problem = EstimationProblem(
    state_preparation=european_call,  
    objective_qubits=[3],
    post_processing=european_call_objective.post_processing,
)

# construct amplitude estimation
ae = IterativeAmplitudeEstimation(
    epsilon_target=epsilon, alpha=alpha, sampler=sampler
)



```


```python
result = ae.estimate(problem)
```


```python
conf_int = np.array(result.confidence_interval_processed)
print("Exact value:        \t%.4f" % exact_value)
print("Estimated value:    \t%.4f" % (result.estimation_processed))
print("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))
```

    Exact value:        	0.1623
    Estimated value:    	0.1650
    Confidence interval:	[0.1585, 0.1714]



```python
from qiskit_finance.applications.estimation import EuropeanCallPricing

european_call_pricing = EuropeanCallPricing(
    num_state_qubits=num_uncertainty_qubits,
    strike_price=strike_price,
    rescaling_factor=c_approx,
    bounds=(low, high),
    uncertainty_model=uncertainty_model,
)
```


```python
# set target precision and confidence level
epsilon = 0.01
alpha = 0.05
sampler = StatevectorSampler()

problem = european_call_pricing.to_estimation_problem()
# construct amplitude estimation
ae = IterativeAmplitudeEstimation(
    epsilon_target=epsilon, alpha=alpha, sampler=sampler
)
result = ae.estimate(problem)

conf_int = np.array(result.confidence_interval_processed)
print("Exact value:        \t%.4f" % exact_value)
print("Estimated value:    \t%.4f" % (european_call_pricing.interpret(result)))
print("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))
```

    Exact value:        	0.1623
    Estimated value:    	0.1674
    Confidence interval:	[0.1587, 0.1761]



```python

from qiskit_finance.applications.estimation import EuropeanCallDelta

european_call_delta = EuropeanCallDelta(
    num_state_qubits=num_uncertainty_qubits,
    strike_price=strike_price,
    bounds=(low, high),
    uncertainty_model=uncertainty_model,
)
```


```python
european_call_delta._objective.decompose().draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">         ┌───────────────┐
state_0: ┤0              ├
         │               │
state_1: ┤1              ├
         │               │
state_2: ┤2              ├
         │  circuit-5483 │
state_3: ┤3              ├
         │               │
 work_0: ┤4              ├
         │               │
 work_1: ┤5              ├
         └───────────────┘</pre>




```python
european_call_delta_circ = QuantumCircuit(european_call_delta._objective.num_qubits)
european_call_delta_circ.append(uncertainty_model, range(num_uncertainty_qubits))
european_call_delta_circ.append(
    european_call_delta._objective, range(european_call_delta._objective.num_qubits)
)

european_call_delta_circ.draw()
```




<pre style="word-wrap: normal;white-space: pre;background: #fff0;line-height: 1.1;font-family: &quot;Courier New&quot;,Courier,monospace">     ┌───────┐┌──────┐
q_0: ┤0      ├┤0     ├
     │       ││      │
q_1: ┤1 P(X) ├┤1     ├
     │       ││      │
q_2: ┤2      ├┤2     ├
     └───────┘│  ECD │
q_3: ─────────┤3     ├
              │      │
q_4: ─────────┤4     ├
              │      │
q_5: ─────────┤5     ├
              └──────┘</pre>




```python
# set target precision and confidence level
epsilon = 0.01
alpha = 0.05
sampler = StatevectorSampler()

problem = european_call_delta.to_estimation_problem()

# construct amplitude estimation
ae_delta = IterativeAmplitudeEstimation(
    epsilon_target=epsilon, alpha=alpha, sampler=sampler
)
```


```python
result_delta = ae_delta.estimate(problem)
```


```python
conf_int = np.array(result_delta.confidence_interval_processed)
print("Exact delta:    \t%.4f" % exact_delta)
print("Estimated value: \t%.4f" % european_call_delta.interpret(result_delta))
print("Confidence interval: \t[%.4f, %.4f]" % tuple(conf_int))
```

    Exact delta:    	0.8098
    Estimated value: 	0.8096
    Confidence interval: 	[0.8066, 0.8127]

