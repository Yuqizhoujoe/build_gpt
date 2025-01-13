# %%
import math
import numpy as np
import matplotlib.pyplot as plt

# %%
# The function f(x) represents the equation: f(x) = 3x^2 - 4x + 5
def f(x):
      return 3*x**2 - 4*x + 5

# %%
f(3.0)

# %%
xs = np.arange(-5,5,0.25)
ys = f(xs)
plt.plot(xs,ys)
# %%
'''
How to calculate the slope of the function at a specific point?

The derivative of a function f at a point x is defined as the limit of the average rate of change of the function over an interval as the interval approaches zero. Mathematically, it is represented as:

f'(x) = lim (h -> 0) [f(x + h) - f(x)] / h

In this case, we approximate the derivative by using a small value for h.
'''
def calculate_slope(x1, x2):
      return (f(x2) - f(x1)) / (x2 - x1)

# %%
h = 0.001
x = 3.0
calculate_slope(x, x+h)

# %%
h = 0.001

# inputs
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
a += h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2-d1)/h)

# %%
class Value:
      def __init__(self, data, _children=(), _op='', label=''):
            self.data = data
            self._prev = set(_children)
            self._op = _op
            self.label = label
            self.grad = 0.0
            # for leaf node: nothing to do
            self._backward = lambda: None
      
      def __repr__(self):
            return f"Value(data={self.data})"
      
      def print_children(self, level=0):
            indent = "  " * level
            print(f"{indent}{self}")
            for child in self._prev:
                  child.print_children(level + 1)
                        
      def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), '+')
            
            '''
            https://en.wikipedia.org/wiki/Chain_rule#Multivariable_case
            
            the reason: self.grad += 1.0 * out.grad?
            prevent gradient from being overwritten
            regarding multi variable case in chain rule, variable gradient should be accumulated when variable is used in multiple parent nodes
            '''
            def _backward():
                  self.grad += 1.0 * out.grad
                  other.grad += 1.0 * out.grad
                  
            out._backward = _backward
            return out
      
      def __neg__(self): # -self
            return self * -1
      
      def __sub__(self, other): # self - other
            return self + (-other)
      
      def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), '*')
            
            '''
            why self.grad += other.data * out.grad?
            refer to: https://en.wikipedia.org/wiki/Chain_rule
            
            why other.data * out.grad?
            f = x * y
            ((x+h) * y - x * y)/h -> y
            df/dx = y
            '''
            def _backward():
                  self.grad += other.data * out.grad
                  other.grad += self.data * out.grad
            
            out._backward = _backward
            return out
      
      def __pow__(self, other): # self ** other
            assert isinstance(other, (int, float)), "only supporting int/float powers for now"
            out = Value(self.data**other, (self, ), f'**{other}')
            
            '''
            why self.grad += other * self.data**(other-1) * out.grad?
            refer to: https://en.wikipedia.org/wiki/Power_rule#Statement_of_the_power_rule
            f = x^y
            df/dx = y * x^(y-1)
            '''
            def _backward():
                  self.grad += other * self.data**(other-1) * out.grad
                  
            out._backward = _backward
            return out

      def __rmul__(self, other): # other * self
            return self * other
      
      def __truediv__(self, other): # self / other
            return self * other**-1
      
      # https://en.wikipedia.org/wiki/Hyperbolic_functions
      def tanh(self):
            x = self.data
            t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
            out = Value(t, (self, ), 'tanh')
            
            def _backward():
                  self.grad += (1 - t**2) * out.grad
            
            out._backward = _backward
            return out
      
      def exp(self):
            x = self.data
            out = Value(math.exp(x), (self, ), 'exp')
      
            '''
            derivative of e^x is e^x
            here out.data = e^x
            '''      
            def _backward():
                  self.grad += out.data * out.grad
                  
            out._backward = _backward
            return out
      
      # topology sort
      def backward(self):
            topo = []
            visited = set()
            
            def build_topo(v):
                  if v not in visited:
                        visited.add(v)
                        for child in v._prev:
                              build_topo(child)
                        # until finish all its child then process v
                        topo.append(v)
            
            # build topo sort tasks 
            build_topo(self)
            
            self.grad = 1.0
            '''
            why reversed?
            in the topo sort, the root node will be the last one to be processed
            so reverse the topo sort then process the root node first
            '''
            for node in reversed(topo):
                  node._backward()
            pass
            
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d*f; L.label = 'L'
L
# %%
from graphviz import Digraph

'''
DFS to build a graph of all nodes and edges
'''
def trace(root):
      # builds a set of all nodes and edges in a graph
      nodes, edges = set(), set()
      def build(v):
            if v not in nodes:
                  print(f"Adding node: {v}")
                  nodes.add(v)
                  for child in v._prev:
                        print(f"Adding edge: {child} -> {v}")
                        edges.add((child, v))
                        build(child)
            print(f"Finished building graph for Node: {v}")
            
      build(root)
      return nodes, edges

def draw_dot(root):
      dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
      
      nodes, edges = trace(root)
      for n in nodes:
            uid = str(id(n))
            # for any value in the graph, create a rectangular ('record') node for it
            # print(f"n.label: {n.label}")
            dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
            if n._op:
                  # if this value is a result of some operation, create a new node for the operation
                  dot.node(name = uid + n._op, label = f"{n._op}")
                  # and connect this node to the input values
                  # dot.edge(source, target)
                  # Value(data=4.0)+ -> Value(data=4.0)
                  # print(f"{n}{n._op} -> {n}")
                  dot.edge(uid + n._op, uid)
      for n1, n2 in edges:
            # connect the output node to the input node
            # n1 is child, n2 is parent
            # Value(data=10.0) -> Value(data=4.0)+
            # print(f"{n1} -> {n2}{n2._op}")
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
            
      return dot

draw_dot(L)
# %%
'''
Verification: get gradient for variables
'''

'''
L = d * f

dL/dd = f -> -2

proof process:
(f(x+h)-f(x))/h
((d+h)*f - d*f)/h
(d*f + h*f - d*f) / h
(h*f)/h
f
'''
f.grad = 4.0
d.grad = -2
L.grad = 1.0

'''
(f(x+h) - f(x)) /h 

((c+h + e) - (c+e))/h
(c + h + e - c - e)/h
h/h
1.0

dd / dc = 1.0
dd / de = 1.0

d = c + e

Derivative chain rule:
WANT:
dL / dc = (dL / dd) * (dd / dc)

KNOW:
dL / dd = f -> -2
dd / dc = 1
'''
c.grad = -2.0
e.grad = -2.0

'''
dL / dd = -2
dd / de = 1
dL / de = -2 = (dL / dd) * (dd / de)

e = a * b
de / da = b -> -3
de / db = a -> 2

dL / da = (dL / de) * (de / da) = -2 * -3 = 6
dL / db = (dL / de) * (de / db) = -2 * 2 = -4
'''
a.grad = 6.0
b.grad = -4.0
draw_dot(L)

# %%
'''
0.01: learning rate or step size
'''
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

e = a * b
d = e + c
L = d * f
print(L.data)

# %%
plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid();

# %%
# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
# b = Value(8, label='b')
b = Value(6.8813735, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b; n.label = 'n'
# activation function
o = n.tanh(); o.label = 'o'
o.backward()

# %%
""" Calculation of gradients
o.grad = 1
o.data = 0.7071

o = tanh(n)

do/dn = 1 - o.data**2
refer to: https://en.wikipedia.org/wiki/Hyperbolic_functions#Derivatives

so n.grad = 1 - o.data**2 = 1 - 0.7071**2 = 1 - 0.5 = 0.5

dn / d(x1w1x2w2) = (x1w1x2w2 + h + b - (x1w1x2w2 + b))/h = h/h = 1
x1w1x2w2.grad = do/d(x1w1x2w2) = do / dn * dn / d(x1w1x2w2) -> 0.5 * 1.0 = 0.5

b.grad = do/db = do/dn * dn/db = 0.5 * 1 = 0.5

x1w1.grad = do/dx1w1 = do/dn * dn/dx1w1x2w2 * dx1w1x2w2/dx1w1 = 0.5 * 1 * 1 = 0.5
x2w2.grad = do/dx2w2 = do/dn * dn/dx1w1x2w2 * dx1w1x2w2/dx2w2 = 0.5 * 1 * 1 = 0.5

dx2w2 / dx2
((x2 + h) * w2 - x2 * w2)/h
(x2 * w2 + h * w2 - x2 * w2)/h
h * w2 / h
w2 = 1

dx2w2 / dw2
((w2+ h) * x2 - w2 * x2)/h
(w2 * x2 + h * x2 - w2 * x2)/h
h * x2 / h
x2 = 0

dx1w1 / dx1
((x1 + h) * w1 - x1 * w1)/h
(x1 * w1 + h * w1 - x1 * w1)/h
h * w1 / h
w1 = -3

dx1w1 / dw1
((w1 + h) * x1 - w1 * x1)/h
(w1 * x1 + h * x1 - w1 * x1)/h
h * x1 / h
x1 = 2

do/dx2 = do/dn * dn/dx1w1x2w2 * dx1w1x2w2/dx2w2 * dx2w2/x2 = 0.5 * 1 * 1 * 1 = 0.5
do/dw2 = do/dn * dn/dx1w1x2w2 * dx1w1x2w2/dx2w2 * dx2w2/dw2 = 0.5 * 1 * 1 * 0 = 0
do/dx1 = do/dn * dn/dx1w1x2w2 * dx1w1x2w2/dx1w1 * dx1w1/dx1 = 0.5 * 1 * 1 * -3 = -1.5
do/dw1 = do/dn * dn/dx1w1x2w2 * dx1w1x2w2/dx1w1 * dx1w1/dw1 = 0.5 * 1 * 1 * 2 = 1
"""
draw_dot(o)
# %%
import torch

x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.8813735]).double(); b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)
print(o.data.item())
o.backward()

print('----')
print('x1.grad', x1.grad.item())
print('w1.grad', w1.grad.item())
print('x2.grad', x2.grad.item())
print('w2.grad', w2.grad.item())
# %%
torch.Tensor([[1,2,3], [4,5,6]])
# %%
'''
nin: number of inputs
nout: number of outputs
sz: size of the network
'''
import random
class Neuron:
      def __init__(self, nin):
            print(f"Neuron initialized with parameters: nin={nin}")
            self.weights = []
            for _ in range(nin):
                  self.weights.append(Value(random.uniform(-1,1)))
            self.bias = Value(random.uniform(-1,1))
            print(f"weights: {self.weights}")
            print(f"bias: {self.bias}")
            
      def __repr__(self) -> str:
            return f"Neuron(weights={self.weights}, bias={self.bias})"
            
      def __call__(self, inputs):
           # w * x + b
           #      wx = [wi*xi for wi, xi in zip(self.w, x)]
           wx_sum = Value(0)
           for wi, xi in zip(self.weights, inputs):
                 wx_sum += wi * xi
                 xi_data = xi.data if isinstance(xi, Value) else xi
                 print(f"{wi.data:.4f} * {xi_data:.4f} = {wi.data * xi_data:.4f}")
                
           print(f"wx_sum: {wx_sum.data:.4f}")
           
           act = wx_sum + self.bias.data
           out = act.tanh()
           print(f"sum({wx_sum.data:.4f}) + {self.bias.data:.4f} = {act.data:.4f}")
           print(f"tanh({act.data:.4f}) = {out.data:.4f}")
           print("----linear transformation & activation----")
           return out
     
      def parameters(self):
            params = self.weights + [self.bias]
            print(f"Neuron parameters: {params}")
            return params

class Layer:   
      def __init__(self, nin, nout):
            print(f"Layer initialized with parameters: num_inputs={nin}, num_outputs={nout}")
            self.neurons = []
            for _ in range(nout):
                  self.neurons.append(Neuron(nin))
            print("\n")
            
      def __call__(self, inputs):
            outs = []
            print(f"Layer call with input: {inputs}")
            print(f"Layer neurons: {self.neurons}")
            for neuron in self.neurons:
                  outs.append(neuron(inputs))
            
            if len(outs) == 1:
                  return outs[0]   
            return outs

      def parameters(self):
            params = []
            for neuron in self.neurons:
                  ps = neuron.parameters()
                  params.extend(ps)
            print(f"Neuron parameters in Layer: {params}")
            return params
            
class MLP:
      def __init__(self, nin, nouts):
            sz = [nin] + nouts
            print(f"MLP initialized with parameters: nin={nin}, nouts={nouts}, sz={sz}\n")
            # self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
            self.layers = []
            for i in range(len(nouts)):
                  self.layers.append(Layer(sz[i], sz[i+1]))
            
            # for i, layer in enumerate(self.layers):
            #       print(f"Layer {i}: {layer}")

      def __call__(self, x):
            print(f"MLP call with input: {x}")
            for layer in self.layers:
                  x = layer(x)
                  print(f"MLP layer(x) result: {x}\n")
            return x

      def parameters(self):
            params = []
            for layer in self.layers:
                  ps = layer.parameters()
                  params.extend(ps)
            print(f"Neuron parameters for all layers in MLP: {len(params)} {params}")
            return params

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)
# %%
n.parameters()

# %%
draw_dot(n(x))

# %%
xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# %%
for k in range(20):
      
      # forward pass
      y_pred = [n(x) for x in xs]
      loss = Value(0)
      for ygt, yout in zip(ys, y_pred):
            loss += (yout - ygt)**2
      
      # backward pass
      for p in n.parameters():
            p.grad = 0.0
      loss.backward()
      
      # update
      # parameter: weights & bias
      for p in n.parameters():
            p.data += -0.05 * p.grad
            
      print(f"epoch {k}: loss {loss.data}\n")

# %%
y_pred
# %%
'''
minimize loss
yout: predicted output
ygt: ground truth
'''
# loss = sum((yout - ygt)**2 for ygt, yout in  zip(ys, y_pred))
# loss
loss = Value(0)
for ygt, yout in zip(ys, y_pred):
      print(f"ygt: {ygt}, yout: {yout}")
      loss += (yout - ygt)**2
loss

# %%
y_pred  = [n(x) for x in xs]
loss = Value(0)
for ygt, yout in zip(ys, y_pred):
      print(f"ygt: {ygt}, yout: {yout}")
      loss += (yout - ygt)**2
loss

# %%
loss.backward()
# %%
for p in n.parameters():
      p.data += -0.01 * p.grad
# %%
y_pred
# %%
