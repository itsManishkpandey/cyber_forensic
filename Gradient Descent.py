cur_x=3
rate=0.01
precision=0.000001
previous_step_size=1
max_iters =10000
iters =0
df=lambda x: 2*(x+5)

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x
    cur_x=cur_x-rate * df(prev_x)
    previous_step_size=abs(cur_x-prev_x)
    iters=iters+1
    print("Iteration",iters,"\nX Value is :",cur_x)

print("The local minimum occurs at",cur_x)

import numpy as np
import matplotlib.pyplot as plt
f_x=lambda x: (x**3)-4*(x**2)+6
x=np.linspace(-1,4,100)
plt.plot(x,f_x(x))
plt.show()
f_x_derivative=lambda x: 3*(x**2) -8*x

def plot_gradient(x,y,x_vis,y_vis):
 plt.subplot(1,2,2)
 plt.scatter(x_vis,y_vis,c='b')
 plt.plot(x,f_x(x),c="r")
 plt.title("Gradient descent")
 plt.show()

 plt.subplot(1,2,1)
 plt.scatter(x_vis,y_vis,c="b")
 plt.plot(x,f_x(x),c="r")
 plt.xlim([2.0,3.0])
 plt.title("Zoomed in figure")
 plt.show()


def gradient_iteration(x_start,iterations,learning_rate):
  x_grad=[x_start]
  y_grad=[f_x(x_start)]
 
  for i in range(iterations):

    x_start_derivative=-f_x_derivative(x_start)
    x_start++(learning_rate*x_start_derivative)
    x_grad.append(x_start)
    y_grad.append(f_x(x_start))
    print("local minimum occurs at: {:.2f}".format(x_start))
    print("number of steps:",len(x_grad)-1)
    plot_gradient(x,f_x(x),x_grad,y_grad)

gradient_iteration(0.5,1000,0.05)
