import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return -y**2 + np.sin(x) # Define the non-linear equation

def adams_bashforth(f, x0, y0, h, n):
    x = [x0]
    y = [y0]
    for i in range(0, n-1):
        # Use fourth-order Runge-Kutta to get the next value of y
        k1 = h*f(x[i], y[i])
        k2 = h*f(x[i] + 0.5*h, y[i] + 0.5*k1)
        k3 = h*f(x[i] + 0.5*h, y[i] + 0.5*k2)
        k4 = h*f(x[i] + h, y[i] + k3)
        y_next = y[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

        # Use Adams-Bashforth to get the next value of y
        if i < 3:
            y_next = y_next # Use fourth-order Runge-Kutta for the first three steps
        else:
            y_next = y[i] + (h/24)*(55*f(x[i], y[i]) - 59*f(x[i-1], y[i-1]) + 37*f(x[i-2], y[i-2]) - 9*f(x[i-3], y[i-3]))

        # Append x and y to their respective lists
        x.append(x[i] + h)
        y.append(y_next)
    return x, y

# Define initial values and parameters
x0 = 0
y0 = 1
h = 0.1
n = 100

# Solve the non-linear equation using Adams-Bashforth method
x, y = adams_bashforth(f, x0, y0, h, n)

# Plot the solution
plt.plot(x, y, label='Adams-Bashforth')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

