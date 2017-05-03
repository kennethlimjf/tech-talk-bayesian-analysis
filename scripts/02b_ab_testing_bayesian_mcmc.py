"""
AB Testing: Bayesian Way (MCMC)
"""

import numpy as np
import pymc3 as pm
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

def generate_web_data(visits, mean):
    return bernoulli.rvs(mean, size=visits)

ctr_a, visits_a = 110, 500
ctr_b, visits_b = 520, 1000

p_a = ctr_a / visits_a
p_b = ctr_b / visits_b

data_a = generate_web_data(visits_a, p_a)
data_b = generate_web_data(visits_b, p_b)

ab_test = pm.Model()
with ab_test:
    theta_a = pm.Beta('theta_a', 1, 1)
    theta_b = pm.Beta('theta_b', 1, 1)

    theta_diff = pm.Deterministic('theta_diff', theta_b - theta_a)

    y1 = pm.Bernoulli('y1', p=theta_a, observed=data_a)
    y2 = pm.Bernoulli('y2', p=theta_b, observed=data_b)

    trace = pm.sample(5000, pm.Metropolis())

pm.traceplot(trace, varnames=['theta_a', 'theta_b'])
pm.traceplot(trace[500:], varnames=['theta_a', 'theta_b'])
pm.plot_posterior(trace[500:], varnames=["theta_a", "theta_b"])


pm.plot_posterior(trace[500:], varnames=["theta_diff"])
