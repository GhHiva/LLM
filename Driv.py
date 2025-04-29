import math
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
# import graphviz
# from graphviz import Digraph




## Function with one variable:
### log function ###
def f(x):
    return np.log(x+2)

### Plot f(x) ###
x= np.arange(-1,10,0.5)
fig,ax = plt.subplots()
ax.plot(x,f(x), marker= 'o',color= 'green', linestyle='--')
ax.set_title('Function log(x+2)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
#plt.show()

### Derivative:
# If h=0.0001 and x=3 then find f(x), f(x+h), f(x+h)-f(x), (f(x+h)-f(x))/h
h , x = 0.0001, 3
print(f" f(x) : {f(x)} \n f(x+h) : {f(x+h)} \n f(x+h)-f(x) : {f(x+h)-f(x)} \n (f(x+h)-f(x))/h : {(f(x+h)-f(x))/h} ")

## Function with three variables:
def g(x1,x2,x3):
    return (x1 * x2 + x3)

# If h=0.0001, x1=3, x2=5, and x3=4 then find g(X), g(X+h), (g(X+h)-g(X))/h
h, x1, x2, x3 =0.0001, 3, 5, 4

### Derivative respect to x1:
print(f" g(X) : {g(x1,x2,x3)} \n g(x1+h,x2,x3) : {g(x1+h,x2,x3)} \n Slope in x1 : {(g(x1+h,x2,x3)-g(x1,x2,x3))/h} ")
#
# ## Derivative respect to x2:
print(f" g(X) : {g(x1,x2,x3)} \n g(x1,x2+h,x3) : {g(x1,x2+h,x3)} \n Slope in x2 : {(g(x1,x2+h,x3)-g(x1,x2,x3))/h} ")
#
# ## Derivative respect to x3:
print(f" g(X) : {g(x1,x2,x3)} \n g(x1,x2,x3+h) : {g(x1,x2,x3+h)} \n Slope in x3 : {(g(x1,x2,x3+h)-g(x1,x2,x3))/h} ")


