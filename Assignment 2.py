#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def f (x):
    """Defines the cubic function f(x)=x^3 - x^2 - 1
    Parameters:
    x ------ array like, numerical values to evaluate f(x)
    Returns:
    f ------ array like, the function f(x)"""
    f = x**3 - x**2 -1
    return f

def df (x):
    """Defines the derivative of the function f(x), df(x) = 3x - 2x
    Parameters:
    x ------ array like, numerical values to evaluate the derivative of f(x)
    Returns:
    df ------ array like, the function df(x)"""
    df = 3*x**2 - 2*x
    return df

def newton(f, df, x0, epsilon= 1e-6, max_iter=30):
    """Defines the Newton iteration of the funtcion f and its derivative df
    Parameters:
    f ----- array like, output of f(x)
    df ----- array like, output of df(x)
    x0 ----- integer like, value of initial x
    """
    
    # Creates an array that contains the different values of xn
    # The loop appends the new x value after each iteration
    xn_array = [x0]
    
    # Iterates at maximum 30 times over the recursive formula
    for i in range(max_iter):
        # Recursive formula
        x = xn_array[i] - (f(xn_array[i])/df(xn_array[i]))
        
        # Appends different x values to our array
        xn_array.append(x)
        
        # Condition to stop the loop
        # Checks if the absolute value of the function f(x) is smaller than epsilon
        if np.abs(f(xn_array[i])) < epsilon:
            break
            
    # Checks if the program could not find a zero of the function
    if len(xn_array) > max_iter:
        print ("Iteration failed")
        return None
    
    # Displays the number of iterations necessary to find the zero of the function
    print("Found root in ",len(xn_array)," interations")
    
    # Prints the last 
    print("Xn: ", xn_array[len(xn_array)-1])

# Try out different examples
# Example 1
newton(f, df, 0.005, epsilon= 1e-6, max_iter=30)

# Example 2
newton(f, df, 5, epsilon= 1e-6, max_iter=30)

# Changing epsilon value
newton(f, df, 0.005, epsilon= 1e-8, max_iter=30)
newton(f, df, 5, epsilon= 1e-8, max_iter=30)

print("It takes one additional iteration if we change the epsilon value to 1e-8")


# In[ ]:




