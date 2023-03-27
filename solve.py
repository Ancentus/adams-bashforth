import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the function f(y, t)
def f(y, t):
    return 0.2 * y - 0.01 * y**2

# Set the initial conditions
y0 = 6
t0 = 2000
tf = 2020
h = 1

# Compute the number of steps
N = int((tf - t0) / h)

# Create arrays to store the solution
t = np.zeros(N+1)
y = np.zeros(N+1)
t[0] = t0
y[0] = y0

# Use the Taylor method to compute the first step
y[1] = y[0] + h * f(y[0], t[0]) + h**2 / 2 * (f(y[0], t[0]) - 0.2 * y[0])
t[1] = t[0] + h

# Use the Adams-Bashforth method to compute the remaining steps
for i in range(2, N+1):
    # Predictor step
    y_pred = y[i-1] + h * (3/2 * f(y[i-1], t[i-1]) - 1/2 * f(y[i-2], t[i-2]))
    t_pred = t[i-1] + h
    
    # Corrector step
    y[i] = y[i-1] + h * (5/12 * f(y_pred, t_pred) + 2/3 * f(y[i-1], t[i-1]) - 1/12 * f(y[i-2], t[i-2]))
    t[i] = t[i-1] + h

# Create a table of the numerical approximation
data = {'time t_i': t, 'Adams approx of non-linear y': y}
df = pd.DataFrame(data)
print(df)

# Plot the solution
plt.plot(t, y, label='Adams-Bashforth Method')
plt.title('Non Linear Population Equation')
plt.legend(loc='best')
plt.xlabel('time (yrs)')
plt.ylabel('Population in billions')
plt.show()

