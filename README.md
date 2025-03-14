# MNIST Classifier
This is a simple implementation of a MNIST classifier from scratch in C++

## Architecture
It leverages a very simple architecture.

Input Layer: 782 neurons
Hidden layer: 512 neurons
Output layer: 10 neurons

- This architecture was chosen because the input numbers are 28 * 28 = 782 pixels.
- The hidden layer is what they use in the book "Deep learning with Python".
- The output layer is because it's supposed to classify the input pixels to a number between 0-9.
- Optimizer is RMSProp, which is also used in the book Deep learning with Python.
- Batch SDG of 32 because it yields the best results.

## Results

Currently the network converges at close to 15 epochs with an accuracy of `~91%`, this is a very bad result.
In the book "Deep learning with Python", they use the Keras framework, and with a similar architecture, using batch size of 128, and only 5 epochs they're able to get above `97%` accuracy. 

I am currently unable to reproduce these results.

My current thought process must be that there is some hyper parameter difference which I am unaware of.

Using true gradient descent, meaning that the batch size is reduced to 1, improves performance. This puts the model at industry standard ~97%.

## Optimizations

There are several C++ optimizations that can be done to increase speed, tremendously, but which are have not been done. This is because I wished to first find a working architecture, and then optimize the code. 

The only optimization that currently really has been thought through is to use flat vectors as arrays, this is because when storing flat memory we can leverage stack usage for `std::vector`. 
This allows all computations to be done on contiguous stack memory, which is severely faster than using heap allocations.

In this case since we only use `std::vectors` with predetermined size and continuous memory blocks we could really use `std::array` as we don't need the dynamic resizing. This would however not yield any performance gains, or very minimal gains, only give a slightly lower memory profile.

### Optimisations to add:
- SIMD vectorisation
- Parallel computation


## Benchmarks
Training time: 282661 ms (282.661 s)

Accuracy (32 batch): 91-92%
Accuracy (Gradient Descent): 97.7%
