# Landing a Rocket with Neuroevolution

**Team Members:** Tyler Garriott, David Long, Gavin Picard

This project serves as our final project for our COSC420/527 course, Introduction to Biologically-Inspired Computing, at The University of Tennessee, Knoxville.

### Research Question
What makes an evolutionary algorithm reliably learn safe, fuel-efficient 2D rocket landings?

### Description
This project investigates how the hyperparameters of a genetic algorithm affect its ability to evolve a neural network that produces control policies for a 2D rocket landing task. A physics simulation with gravity, drag, wind, and fuel constraints provides the environment. Each genome encodes the weights of a feedforward neural network (6 → 16 → 16 → 2); the GA evolves this population using tournament selection, uniform crossover, Gaussian mutation, and elitism. A novelty search bonus encourages behavioral diversity. A curriculum gradually ramps up wind and physics variation across training. We vary the GA parameters and measure their effect on fitness convergence, landing success rate, and fuel efficiency.

### Parameters Under Investigation (Default Values)
- POPULATION_SIZE = 100
- NUM_GENERATIONS = 300
- TOURNAMENT_SIZE = 7
- CROSSOVER_RATE = 0.7
- MUTATION_RATE = 0.1
- MUTATION_SIGMA = 0.1
- ELITISM_COUNT = 5
