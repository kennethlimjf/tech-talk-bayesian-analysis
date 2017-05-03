%matplotlib inline

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import theano

size = 100
true_intercept = 1
true_slope = 2

# Prepare data
x = np.linspace(0, 1, size)
true_regression_line = true_intercept + true_slope * x
y = true_regression_line + np.random.normal(scale=.5, size=size)
x_out = np.append(x, [.1, .15, .2])
y_out = np.append(y, [8, 6, 9])
data = dict(x=x_out, y=y_out)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel='x', ylabel='y', title='Generated data and underlying model')
ax.plot(x_out, y_out, 'x', label='sampled data')
ax.plot(x, true_regression_line, label='true regression line', lw=2.)
plt.legend(loc=0)



# Normal Linear regression
with pm.Model() as model:
    pm.glm.glm('y ~ x', data)
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace = pm.sample(2000, step, progressbar=False)

plt.subplot(111, xlabel='x', ylabel='y', title='Posterior predictive regression lines')
plt.plot(x_out, y_out, 'x', label='data')
pm.glm.plot_posterior_predictive(trace, samples=100, label='posterior predictive regression lines')
plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
plt.legend(loc=0)


# Robust Linear Regression
with pm.Model() as model_robust:
    family = pm.glm.families.StudentT()
    pm.glm.glm('y ~ x', data, family=family)
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_robust = pm.sample(2000, step, progressbar=False)

plt.figure(figsize=(5, 5))
plt.plot(x_out, y_out, 'x')
pm.glm.plot_posterior_predictive(trace_robust, label='posterior predictive regression lines')
plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')
plt.legend()
























#
