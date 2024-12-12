import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


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
            out = Value(self.data + other.data, (self, other), '+')
            
            def _backward():
                  self.grad = 1.0 * out.grad
                  other.grad = 1.0 * out.grad
                  
            out._backward = _backward
            return out
      
      def __mul__(self, other):
            out = Value(self.data * other.data, (self, other), '*')
            
            def _backward():
                  self.grad = other.data * out.grad
                  other.grad = self.data * out.grad
            
            out._backward = _backward
            return out
      
      # https://en.wikipedia.org/wiki/Hyperbolic_functions
      def tanh(self):
            x = self.data
            t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
            out = Value(t, (self, ), 'tanh')
            
            def _backward():
                  self.grad = (1 - t**2) * out.grad
                  print(f'tanh backward: {self.label} {self.grad}')
            
            print(f'out: {out}')
            out._backward = _backward
            return out
            
def trace(root):
      # builds a set of all nodes and edges in a graph
      nodes, edges = set(), set()
      def build(v):
            if v not in nodes:
                  # print(f"Adding node: {v}")
                  nodes.add(v)
                  for child in v._prev:
                        # print(f"Adding edge: {child} -> {v}")
                        edges.add((child, v))
                        build(child)
            # print(f"Finished building graph for Node: {v}")
            
      build(root)
      return nodes, edges

def draw_dot(root):
      dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
      
      nodes, edges = trace(root)
      # print(f"Nodes: {nodes}")
      # print(f"Edges: {edges}")
      for n in nodes:
            uid = str(id(n))
            # for any value in the graph, create a rectangular ('record') node for it
            # print(f"n.label: {n.label}")
            dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
            if n._op:
                  # if this value is a result of some operation, create a new node for the operation
                  dot.node(name = uid + n._op, label = n._op)
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
y =  Value(2, label='y')
z = o*y; z.label='z'; z.grad = 1.0

z._backward()
o._backward()

# n._backward()
# b._backward()
# x1w1x2w2._backward()
# x2w2._backward()
# x1w1._backward()

# draw_dot(z)