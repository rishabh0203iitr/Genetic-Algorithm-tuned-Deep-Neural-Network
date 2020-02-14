# Genetic Algorithm tuned Deep Neural Network

Using Genetic Algorithm to optimize the weights in a deep neural network for classification on MNIST dataset

### Prerequisites

Before running the code install the following python libraries

* numpy
* pymoo
* deap

```
pip install numpy
pip install pymoo
pip install deap
```

### Training the Model

The neural network is developed entirely by using only numpy library. The initial population of the weights are generated randomly, and the fitness of all the models is calculated. The best population is selected according to the fitness function and crossover and mutation is susequntly applied to get new population.


### Results

The Neural Network reaches 60% training accuracy in about 600 generations, which is a very good
result considering the initial population size is just 20.
If the population size is increased to about 100, we may get close to 90% training accuracy.

### Authors

* **Yash Vardhan Sharma** 
* **Rishabh Sharma**

### Acknowledgments

* Ahmed Ghad
* Ali Ghodsi
