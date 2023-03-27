import numpy as np
import matplotlib.pyplot as plt

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

# Use the Runge-Kutta method to compute the first step
k1 = h * f(y[0], t[0])
k2 = h * f(y[0] + 0.5 * k1, t[0] + 0.5 * h)
k3 = h * f(y[0] + 0.5 * k2, t[0] + 0.5 * h)
k4 = h * f(y[0] + k3, t[0] + h)
y[1] = y[0] + (k1 + 2*k2 + 2*k3 + k4) / 6
t[1] = t[0] + h

# Use the Adams-Bashforth method to compute the remaining steps
for i in range(2, N+1):
    # Predictor step
    y_pred = y[i-1] + h * (3/2 * f(y[i-1], t[i-1]) - 1/2 * f(y[i-2], t[i-2]))
    t_pred = t[i-1] + h
    
    # Corrector step
    y[i] = y[i-1] + h * (5/12 * f(y_pred, t_pred) + 2/3 * f(y[i-1], t[i-1]) - 1/12 * f(y[i-2], t[i-2]))
    t[i] = t[i-1] + h

# Plot the solution
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Population growth over time')
plt.show()

