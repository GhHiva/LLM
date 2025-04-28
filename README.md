# LLM

micrograd_from_scratch: making the Value class :

**Notes :**
- _op : is used for showing the operations (it will be executed in graph. In the Value class, it's just the internal attribute.)
- _children : is used for storing the child of chain (it will be executed in graph. In the Value class, it's just the internal attribute.)
- label : is used for labeling the nodes
- self._backward = lambda: None, It's just the empty function!
- isinstance(other, Value): is used to make sure other.data has Value type.
- assert isinstance(other, (int, float)), 'only supporting int/float powers for now' : is used to emphasize the type and pop up AssertionError 'only supporting int/float powers for now'.
- out._backward = _backward : is used to just store the funcition. We do not want to execute the function!
- backward(self) : is used to collect the nodes


2.makemore: Multilayer Perceptron: strating by two characters and predict the third one (Neural Network)


3.makemore: Multilayer Perceptron: strating by three characters and predict the fourth one (Neural Network)
