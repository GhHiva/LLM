import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad=0.0 # initialize the gradient
        self._op= _op #show the operation in the graph
        self._prev= set(_children) # _name: is for keeping the result and updating based on the previous step.
        self._backward = lambda: None
        self.label = label  # here we wanna labeling the numbers whoes we work on

    def __repr__(self): # represent the output readable
        return(f'Value(data={self.data})')

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        #print(f'This is addition : {out}')
        return out #return the output as the Value not just the plain number # It helps to just show the result without None statement

    def __neg__(self): # without this the code does not underestand the -2! I think since the sub works on the right side not left! in the case we just have one number!
        return self * -1


    def __sub__(self, other):
        out = self.data + ( - other )
        #print(f'This is sub : {out}')
        return Value(out)

    def __mul__(self, other):
        other = other if isinstance (other, Value) else Value(other)
        out = Value (self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward= _backward
        #print(f'This is mul : {out}')
        return out

    def __rmul__(self, other): # commuting the product. Multiplication does not always commute! Like Matrix!
        return self * other


    def __truediv__(self, other): #python 3 use __truediv__ instead of __div__
        out = self.data / other
        # print(f'This is div : {out}')
        return Value(out) # we can just use the return

    def __pow__(self, other):
        out = Value (self.data ** other, (self,), f'^ {other}')
        #print(f'This is pow : {out}')
        def _backwrd():
            self.grad += out.grad * (other)* (self.data ** (other -1))
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value (math.exp(x), (self,), 'exp')
        def _backward():
          self.grad += out.grad * out.data
        out._backward = _backward
        return out


    def tanh(self):
        x = self.data
        t = (1-math.exp(2*x))/(1+math.exp(2*x))
        out = Value(t, (self,), 'tanh') # (self,) is a child of tanh
        def _backward():
            self.grad += out.grad * (1-t**2)
        out._backward = _backward # here we save function we do not want to execute it now so we write it without () !

        return out


    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1
        for node in reversed(topo):
            node._backward()


###### Drawing #####

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes, edges



def draw_dot(root):
    dot = Digraph(format = 'png', graph_attr={'rankdir': 'LR'})
    nodes, edges= trace(root)
    for n in nodes:
        uid=str(id(n))
        dot.node(name = uid , label= f"{{{n.label}| data {n.data: .4f} | grad {n.grad: .4f}}}", shape='record')
        if n._op:
            dot.node(name = uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1,n2 in edges:
        dot.edge(str(id(n1)),str(id(n2))+ n2._op)
    return dot



class Neuron:
    def __init__(self, nin):
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b= Value(random.uniform(-1,1))

    def __call__(self, x):
        # w*x+b
        #sum(iterable, start=0)
        a = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out= a.tanh()
        return out
        
    def para(self):
        return self.w+[self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neu= [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out= [n(x) for n in self.neu]
        return (out[0] if len(out)==1 else out) # at first I just returned the `out` but I found that when it is just one number it shows it as a list and we could not in loss function use the subtraction of list and float or int!

        def para(self):
        return [ p for n in self.neu for p in n.para()]

class MLP:

    def __init__(self, nin, nouts):
        s= [nin]+ nouts
        self.layer =[Layer(s[i],s[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layer:
            out =layer(x)
        return out


    def para(self):
        return [p for l in self.layer for p in l.para()]







if __name__=='__main__':
    print('Yes')
