# LLM


## MakeMore1:

### micrograd_from_scratch:

1. Creating the **Value class** :

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

## Drawing the Chain of Calculations:

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

  ## manually insert the gradients:
  
     l.grad= 1 (assumption)
  
     f.grad= 1 ($\frac{d l}{d f} = 1+ \frac{d e}{d f}=1$)
  
     d.grad= 5 ($\frac{d l}{d d} = \frac{d f}{d d}+ \frac{d e}{d d}= \frac{d c}{d d} \cdot d + c \cdot \frac{d d}{d d}+0= 5 $)
  
     e.grad= 1
  
     c.grad= -1
  
     b.grad= 5
  
     a.grad= 5

     draw_dot(l).render('ChainWithGrad', cleanup=True)

     Output: ![ChainWithGrad](https://github.com/user-attachments/assets/85526321-e42a-4f9e-ab12-edad14dbc9be)

  ## inserting gradient by ._backward functions from Value class:
     l.grad=1
  
     l._backward()
  
     f._backward()
  
     d._backward()
  
     e._backward()
  
     c._backward()
  
     b._backward()
  
     a._backward()
  
     draw_dot(l).render('ChainWithGradBack', cleanup=True)
  
     Output: ![ChainWithGradBack](https://github.com/user-attachments/assets/b2962bf9-cffd-417a-94c0-8fe829550fbd)
  ## Needed steps for optimizing:
  
  2. Creating the **Neuron class** :
     
     **Notes:**
     
     - We defined the operations in Value class and how calculating their gradients. Now we need to make the nodes to start by initializing the weights and bias.
     - w is the weight node which its dimensional depends to the input lenght vector.
     - b is a bias node which is one dimensional.
     - w and b are the random vectors which obey of uniform distribution.
     - Finding $`w \cdot x +b = \sum_{i=1}^{i= \text{ dimension }} +b`$.
     - $`\tanh(w \cdot x +b)`$.
     - Neuron(4): has 4 inputs.


  4. Creating the **Layer class** :
     
     **Notes:**
     
     - After creating the nodes, the turn is for making the layers which are pushing the inputs forward to the outputs.
     - Determining the number of outputs we expect from the input.
     - My description:
       
        $`\{(W^{(1)}, b^{(1)}),(W^{(2)}, b^{(2)}), \cdots, (W^{(s)}, b^{(s)})\},`$
       
      each element of above set acts on $`X`$ (the number of outputs we expect is $`s`$). $W^{(i)}$ and $X$ have the same dimension.
      - Layer(4,6): has 4 inputs and 6 outputs for the one layer.

  6. Creating the **MLP class**:

     **Notes:**
     - We have a couple of layers and each layers have neurons. The out put layer of the first calculation is the input layer for the next step.
     - For MLP(4,[6,3,2]), 4 in the dimension of inputs, [6,3,2] means we wanna add three layers, and they have 6, 3, 2 neurons respectively.
     
     

**Loss:** The outputs we find after the MLP can be our desired outputs or not. Since the weights and biases are random numbers, then we have a wide variety of numbers. In this case, we try to find out the predicted output as close as possible to the desired outputs. Means, we wanna reduce the difference between them, or we can say $\lim_t |y_{\text{pred}} - y_{\text{true}}| \to 0$.
, where $t$ is the number of times we try! 
- Loss is the accuracy of the prediction!
- Ex: 

Check the code in Example file : ![loss](https://github.com/user-attachments/assets/623b2924-6ce8-45f9-ad0d-eced390fe45c)

## Steps for Optimizing:
 1. Forward step:
     - Finding the list of y_preds based on the MLP you defined.
     - Finding the loss value ==> loss = sum([(y_p - y_t)**2 for y_p, y_t in zip(y_pred - y_true)],0.0)
 2. Backward step:
     - loss.backward()
 3. Updating step:
     - changing the parameters,

 and repeating these steps until reaching to the desired output which is the minimizing the loss! check the Example3.
 
_________________________________________________________
## MakeMore2: 

- Multilayer Perceptron: strating by two characters and predict the third one (Neural Network)

____________________________________________________________
## MakeMore3: 
- Multilayer Perceptron: strating by three characters and predict the fourth one (Neural Network)
