# Schematics

1.neural_network.py - Implements a fully connected classification neural network with custom architecture, vectorized backpropagation and adaptive learning rate 

# Usage 

A config.txt should be supplied with the following format as a command line argument.

      85
      10
      100
      2
      40 40
      relu
      fixed

Line 1 - Dimension of input parameters  
Line 2 - Units in the final layer  
Line 3 - Batch Size for backpropagation  
Line 4 - Number of hidden layers  
Line 5 - Number of units in each hidden layer space seperated  
Line 6 - Activation Function in the hidden layers. "relu" for RELU and "sigmoid" for sigmoid  
Line 7 - "fixed" for fixed learning rate or "variable" for adaptive learning rate  

Run the code as  
    
    python3 neural_netwok.py config.txt train_data test_data

# Dataset 

[Poker Hand Dataset](https://archive.ics.uci.edu/ml/datasets/Poker+Hand)

# Preprocessing 

One-Hot Encoding
