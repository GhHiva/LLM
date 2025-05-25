## MakeMore2: 

- Multilayer Perceptron: strating by two characters and predict the third one (Neural Network)
  
**Notes:**
  
  🔺 list('emma') = ['e','m','m','a']
  
  🔺 'S' and 'E' act like indicator to show where the start and end of word is.
  
  🔺 b.get(bigram,0): is used to show the values of the keys,. If there is no value it returns 0.
  
  🔺 sorted(b.items): sorts based on the keys. for example ('S', 'a') goes first than ('S', 'e').
  
       - sorted(b.items(), reverse = True): for example ('S', 'e') goes before ('S', 'a').
       
       - sorted(b.items(), key = lambda kv: kv[1], reverse = True) :
       
          - `lambda kv: kv[1]` : means focus on the value and then sort. since reverse is True, the sorting is descending.
          
  🔺`''.join(words)`: sticks all names together

  🔺`set(''.join(words))`: gives the set of unique alphabets that are contained in the ''.join(words)
  
  🔺 Define stoi={s:i for i , s in enumerate(set(''.join(words)))} => `s` is the `letter` and `i` is an `integer` from 0 to 25, here we represent each letter by the integer.
  
  🔺`torch.Generator()` vs `torch.manual_seed(42)`:
    
       - `torch.Generator()`: lets you control random number generation, especially useful when you want reproducibility in your experiments (e.g., for shuffling data or initializing weights). `local generator seeding`.
    
  * Key Points:
         
    - It creates a random number generator that you can seed independently.
         
    - This is helpful when you want to use different seeds for different parts of your code without affecting the global state.
         
    - This creates an `isolated random number generator`, separate from the global one.
         
    - Only functions that explicitly use generator=g are affected by this seed. The global state is untouched, so other calls to `torch.rand() (without a generator) can behave independently`.
         
     Ex: g = torch.Generator()

    g.manual_seed(42) ==> It makes the random behavior repeatable. Every time you run your code with the same seed (42 here), you'll get the same random outputs from that generator.

    rand_tensor = torch.rand(2, 2, generator=g) ==>  Use it in a random operation
         
     - `torch.manual_seed(42) — Global Seed`: This sets the seed for all random number generation in PyTorch `globally`.
    
          Ex: Any function that uses `randomness` (torch.rand, torch.randn, shuffling in DataLoader, etc.) will now give repeatable results `globally`.
 
  🔺 `torch.distributions.Multinomial` vs `torch.multinomial` :
  
  - torch.distributions.Multinomial: A distribution class from torch.distributions. It's object-oriented and supports methods like .sample() and .log_prob(). Use Case: Sample multiple times from a categorical distribution in a single call. Work with probabilistic modeling, including computing log-probabilities.
- torch.multinomial: A function, not a distribution. It samples from a categorical distribution (or draws without replacement), similar to drawing lottery tickets. Use Case: Just need to sample indices according to given probabilities. Don't need log-probabilities or to create a full distribution object.

 🔺 Quality of a model by Likelihood: the high negative likihood the better estimation.
 
