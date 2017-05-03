"""
AB Testing: Bayesian Way (Analytical, Beta-Bernoulli)
"""

import numpy as np
from scipy.stats import beta, bernoulli
from scipy.special import beta as beta_func
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Create web data
def generate_web_data(visits, mean):
    return bernoulli.rvs(mean, size=visits)

ctr_a, visits_a = 220, 500
ctr_b, visits_b = 250, 500

p_a = ctr_a / visits_a
p_b = ctr_b / visits_b

data_a = generate_web_data(visits_a, p_a)
data_b = generate_web_data(visits_b, p_b)


# Beta prior shape params
beta_a = 1
beta_b = 1
prior_shape = [beta_a, beta_b]
thetas = np.linspace(0, 1, 100)
priors = beta.pdf(thetas, *prior_shape)

total_a, total_b = data_a.shape[0], data_b.shape[0]
clicks_a, clicks_b = data_a.sum(), data_b.sum()
non_clicks_a, non_clicks_b = (total_a - clicks_a), (total_b - clicks_b)

# Bayesian update for Beta-Bernoulli
posterior_shape_a = [beta_a + clicks_a, beta_b + non_clicks_a]
posterior_shape_b = [beta_a + clicks_b, beta_b + non_clicks_b]

# Calculate posterior
posterior_a = beta.pdf(thetas, *posterior_shape_a)
posterior_b = beta.pdf(thetas, *posterior_shape_b)

# Plot
plt.figure(figsize=(10,4))
plt.ylabel("$P(\\theta)$")
plt.xlabel("$\\theta$")
plt.plot(thetas, priors, color='blue')
plt.plot(thetas, posterior_a, color='red')
plt.plot(thetas, posterior_b, color='green')

# Calculate the mean conversion rate for a and b
mean_a = float(posterior_shape_a[0]) / float(sum(posterior_shape_a))
mean_b = float(posterior_shape_b[0]) / float(sum(posterior_shape_b))

mean_a
mean_b


# Probability B is better than A
def probability_b_is_better_than_a(a_a, b_a, a_b, b_b):
    total = 0.0
    for i in range(a_b - 1):
        numerator = beta_func(a_a + i, b_a + b_b)
        denominator = (b_b + i) * beta_func(1+i, b_b) * beta_func(a_a, b_a)
        total += (numerator / denominator)
    return total

p = probability_b_is_better_than_a(
    1+clicks_a, 1+non_clicks_a, 1+clicks_b, 1+non_clicks_b)

print("%.10f" % p)































#
