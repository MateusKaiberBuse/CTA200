#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('run', 'complex_function.ipynb')

# Defines point in the complex plane
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)

# Creates a numpy array
complex_set = np.zeros((1000, 1000))

# Creates the complex plane
# Iterates over every x point
for i in range(len(x)):
    # Iterates over every y point
    for j in range(len(y)):
        
        # Defines a complex number for every point in the set
        c = complex(x[j], y[i])
        
        complex_set[i, j] = complex_function(20, c)

# Creates a set that does not account for the number of iterations
# Assumes every divergent points diverged at 1
complex_img = np.copy(complex_set)/complex_set
complex_img[np.isnan(complex_img)] = 0
    
# Displays plot with no number of iterations
plt.imshow(complex_img, extent = [-2, 2, -2, 2], cmap = 'hot')
plt.title("Complex Plane")
plt.xlabel("Re (c)")
plt.ylabel("Im (c)")
plt.show()        

# Displays plot showing the number of iterations
plt.imshow(complex_set, extent = [-2, 2, -2, 2] , cmap = 'hot')
plt.colorbar()
plt.title("Complex Plane")
plt.xlabel("Re (c)")
plt.ylabel("Im (c)")
plt.show()


# # Question 2

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Defines Lorenzs´ parameters and initial conditions
param = [10., 28., 8./3.]
initial_cond = (0., 1., 0.)

def lorenz(t, W_0, param):
    """Defines the system of differential equations, takes a time interval, initial conditions array 3D array, and a
    3D parameter array as input. Outputs the system of differential equations
    Parameters:
    t------array like, time interval
    W_0------array like, initical conditions of the system
    param------array like, parameters of the equation
    
    Output:
    fx, fy, fz------array like, differential equations"""
    
    # Attributes variables to each index of the arrays containing the initial conditions and parameters
    X, Y, Z = W_0
    sigma, r, b = param
    
    # x component
    fx = -sigma*(X-Y)
    
    # y component
    fy = r*X - Y - X*Z
    
    # z component
    fz = -b*Z + X*Y
    
    return fx, fy, fz

# Solves the system of differential equation by integration, here we integrate for t = 60
sol = integrate.solve_ivp(lambda t, W_0: lorenz(t, W_0, param), (0., 60.), initial_cond, t_eval=np.linspace(0, 60, 3000))
x, y, z = sol.y

# Reproducing Lorenz figure 1
# Creates figure with space for all 3 subplots
fig1 = plt.figure(figsize = (5,10))
fig1.suptitle('Figure 1: Numerical solution of the convection equations', fontsize=16)

# Adds subplots to figure 1
ax1 = fig1.add_subplot(3, 1, 1)
# Label subplots axis and title
ax1.set_title('Y in the first thousand iterations')
ax1.set_xlabel("First 1000 Iterations")
ax1.set_ylabel("Y")

ax2 = fig1.add_subplot(3, 1, 2)
# Label subplots axis and title
ax2.set_title('Y in the second thousand iterations')
ax2.set_xlabel("Second 1000 Iterations")
ax2.set_ylabel("Y")

ax3 = fig1.add_subplot(3, 1, 3)
# Label subplots axis and title
ax3.set_title('Y in the third thousand iterations')
ax3.set_xlabel("Third 1000 Iterations")
ax3.set_ylabel("Y")

# Plots subplots
ax1.plot(sol.t[0:500], y[0:500])
ax2.plot(sol.t[500:1000], y[500:1000])
ax3.plot(sol.t[1000:1500], y[1000:1500])

fig1.tight_layout()
plt.show()

# Integrates the system of differential equations but now at a different time interval
sol2 = integrate.solve_ivp (lambda t, W_0: lorenz(t, W_0, param), (0. , 60.), initial_cond, t_eval = np.linspace(14,19, 1000))
x2, y2, z2 = sol2.y

# Reproducing figure 2
# Creates a figure with space for 2 subplots
fig2 = plt.figure(figsize = (5, 8))
fig2.suptitle('Figure 2: Projections on XY plane and YZ plane', fontsize=16)

# Adds subplots to figure 2
ax1 = fig2.add_subplot(2, 1, 1)
ax2 = fig2.add_subplot(2, 1, 2)

# Displays subplots
ax1.plot(y2, z2)
# Labels subplots axis and title
ax1.set_title('Solutions projected on the YZ plane')
ax1.set_xlabel("Y-axis")
ax1.set_ylabel("Z-axis")

ax2.plot(y2, -x2)
# Label subplots axis and title
ax2.set_title('Solutions projected on the XY plane')
ax2.set_xlabel("-X-axis")
ax2.set_ylabel("Y-axis")

fig2.tight_layout()
plt.show()


# In[5]:


# Now changing the initial conditions
new_initial_cond = [0., 1.00000001, 0.]

# Use Lorenz function with new initial conditions to get new set of solutions
sol3 = integrate.solve_ivp(lambda t, W_0: lorenz(t, W_0, param), (0., 60.), new_initial_cond, t_eval=np.linspace(0, 60, 3000))
x3, y3, z3 = sol3.y

# Calculate distanec between solutions
distance = np.sqrt((x3 - x)**2 + (y3 - y)**2 + (z3 - z)**2)

# Displays distance as a function of time
plt.plot(np.log(distance), sol3.t)
plt.title("Distance between two solutions as a function of time")
plt.xlabel("Distance between two solutions (log scale)")
plt.ylabel("Time interval")
plt.show()


# In[ ]:




