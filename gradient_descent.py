import matplotlib.pyplot as plt
import numpy as np

def f(x):
  return (x**2) - (6*x) + 3

def derivative_f(x):
  return (x*2) - 6

def gradient_descent(x):
  stepsize = 0.05
  return x - (stepsize*derivative_f(x))

def drawFunction():
  x = np.linspace(-2,8,100)
  y = f(x)
  plt.plot(x,y, 'r')
  plt.axis([-2, 8, -10, 15])

drawFunction()
x = 10
while(True):
  print('x:', x)
  new_x = gradient_descent(x)
  plt.scatter(x, f(x), c="blue")
  if abs(new_x - x) < 0.0000001:
    break
  x = new_x
plt.show()
print('final x:', x)
