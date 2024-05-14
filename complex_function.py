#!/usr/bin/env python
# coding: utf-8

# In[1]:


def complex_function(max_iter, c):
    """Defines the recursive function z(i+1) = z(i) + c where c is a point on the complex plane
    Parameters:
    max_iter-----integer like, maximum number of iterations that the function will loop
    c-----complex number like, a point on the complex plane of the form c = x + iy
    Output:
    i------integer like, number of iterations that it took for a given point to diverge"""
    
    # Defines initial z value
    z = 0
    
    for i in range(max_iter):
        z = z**2 + c
        
        # Checks if the absolute value of z diverges and if so returns the number of iterations that it took to diverge
        if abs(z) > 2:
            return i
    return max_iter


# In[ ]:




