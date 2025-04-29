# LLM

## micrograd_from_scratch:

1. making the **Value class** :

**Notes :**
- `_op` : is used for showing the operations (it will be executed in graph. In the Value class, it's just the internal attribute.)
- `_children` : is used for storing the child of chain (it will be executed in graph. In the Value class, it's just the internal attribute.)
- `label` : is used for labeling the nodes
- `self._backward = lambda: None`, It's just the empty function!
- `isinstance(other, Value)`: is used to make sure other.data has Value type.
- `assert isinstance(other, (int, float)), 'only supporting int/float powers for now'` : is used to emphasize the type and pop up AssertionError 'only supporting int/float powers for now' if the verdict did not satisfied.
- `out._backward` = _backward : is used to just store the funcition. We do not want to execute the function!
- `backward(self)` : is used to collect the nodes
 
 **Example:**
  
a = Value(2.0,label='a')

b = Value(-3, label= 'b')

q = a+b ; q.label= 'q'

print(f' q is equal to : {q}\n The operation has been used is: {q._op}\n The children of q are: {q._prev}\n The label of q is: {q.label}')

print(f'The data of a and b are: {a.data, b.data}')

Outputs: 

q is equal to : Value(data=-1.0)

The operation has been used is: 

The children of q are: {'+'}

The label of q is: q

The data of a and b are: (2.0, -3)

![Add](https://github.com/user-attachments/assets/816c445c-1210-49a8-ad50-78e80d919d93)

**Drawing the Chain of Calculations:**
**Notes:**
- `str(n)` vs `str(id(n))`:
    - `str(n)` calls the `__str__` or `__repr__` method of the object `n`. In class Value, we defined __repr__ so we have `Value(data=5.0)`. => NOT guaranteed unique (two different nodes with same data will have same string)

    - `id(n)` gives the `memory address of the object`, as an integer.
      
    - `str(id(n))` turns that memory address into a `string`, like: '140435943962768'. => Always unique (for different objects)

      
- `'{%s | data %.4f | grad %.4f}' % (n.label, n.data, n.grad)`: It's an old form of formatting. The new one is `'{{{n.label}| data {n.data:.4f} | grad {n.grad:.4f}}}'`.
   - %s : Insert a string

- `dot.edge(str(id(n))+ n._op, str(id(n)))` vs `dot.edge(n._op, str(id(n)))`:
   - `dot.edge(n._op, str(id(n)))`: n._op is just the symbol not the unique node id. So it needs to specify which node id you want to work on. Without the unique name id for all same operations in the chain we just have one node and it crash the graph!
   - `dot.edge(str(id(n))+ n._op, str(id(n)))`: str(id(n))+ n._op makes the unique id we need to have! It fakes the unique id for each operation by adding the node name. For example:
      - "Value(data=5.0)+" and "Value(data=7.0)+" have the same operation `+` but identified by two different nodes.

  **Example:**
  a = Value(2.0,label='a')
  
  b = Value(-3, label= 'b')
  
  c = Value(5, label= 'c')
  
  e = Value(10, label= 'e')
  
  d = a+b ; d.label= 'd'
  
  f = c*d ; f.label= 'f'
  
  l = f+e ; l.label= 'l'
  
  draw_dot(l)
  
  Output: ![Chain](https://github.com/user-attachments/assets/98bd8e5b-3c49-48cc-b589-57ca97c32b3e)

  

2.makemore: Multilayer Perceptron: strating by two characters and predict the third one (Neural Network)


3.makemore: Multilayer Perceptron: strating by three characters and predict the fourth one (Neural Network)
