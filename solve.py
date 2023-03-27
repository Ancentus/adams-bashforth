import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return -y**2 + np.sin(x) # Define the non-linear equation

def adams_bashforth(f, x0, y0, h, n):
    x = [x0]
    y = [y0]
    for i in range(0, n-1):
        if i == 0:
            # Use first-order Euler method to get the second value of y
            y_next = y[i] + h*f(x[i], y[i])
        elif i == 1:
            # Use second-order Adams-Bashforth to get the third value of y
            y_next = y[i] + (h/2)*(3*f(x[i], y[i]) - f(x[i-1], y[i-1]))
        elif i == 2:
            # Use third-order Adams-Bashforth to get the fourth value of y
            y_next = y[i] + (h/12)*(23*f(x[i], y[i]) - 16*f(x[i-1], y[i-1]) + 5*f(x[i-2], y[i-2]))
        else:
            # Use fourth-order Adams-Bashforth to get the fifth and subsequent values of y
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

