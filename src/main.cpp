#include "mnist.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <ratio>
#include <string>
#include <vector>

using std::vector, std::transform, std::accumulate, std::fill, std::max,
    std::ifstream, std::string, std::cout, std::random_device,
    std::default_random_engine, std::normal_distribution, std::iota,
    std::shuffle, std::mt19937;

// Hyperparameters for RMSProp
// Will refactor these later but for now we can keep them here.
// Want to easily try to change things as we go, currently at 89.07% accuracy
// but going for above 97%
const float decay_rate = 0.9f;
const float epsilon = 1e-7f;

auto benchmark_start() {
    using std::milli, std::chrono::duration, std::chrono::duration_cast,
        std::chrono::high_resolution_clock, std::chrono::milliseconds;

    return high_resolution_clock::now();
}

auto benchmark_end(std::chrono::time_point<std::chrono::steady_clock> t1) {
    using std::milli, std::chrono::duration, std::chrono::duration_cast,
        std::chrono::high_resolution_clock, std::chrono::milliseconds;

    auto t2 = high_resolution_clock::now();
    // Getting number of milliseconds as an integer.
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    // Getting number of milliseconds as a double.
    duration<double, milli> ms_double = t2 - t1;

    cout << ms_int.count() << "ms\n";
    cout << ms_double.count() << "ms\n";
}

struct Model {
    float learningRate;
    vector<float> inputLayer;
    vector<float> hiddenLayer;
    vector<float> outputLayer;
    vector<float> w1;
    vector<float> b1;
    vector<float> w2;
    vector<float> b2;
};

struct Gradients {
    vector<float> dW1;
    vector<float> dB1;
    vector<float> dW2;
    vector<float> dB2;
    vector<float> dHidden;
    vector<float> dOutput;
};

// Activation function that is used to squash the output of the network to a
// probability distribution.
// void softmax(vector<float>& input) {
//    auto sum = accumulate(input.begin(), input.end(), 0.0,
//                          [](float a, float b) { return a + exp(b); });
//
//    auto fraction = 1.0 / sum;
//    transform(input.begin(), input.end(), input.begin(),
//              [sum, fraction](float x) { return exp(x) * fraction; });
//}

void softmax(vector<float>& input) {
    if (input.empty())
        return;

    auto maxVal = *max_element(input.begin(), input.end());
    vector<float> exp_vals(input.size());
    transform(input.begin(), input.end(), exp_vals.begin(),
              [maxVal](float x) { return exp(x - maxVal); });

    auto sum = accumulate(exp_vals.begin(), exp_vals.end(), 0.0f);
    auto fraction = 1.0 / sum;
    transform(exp_vals.begin(), exp_vals.end(), input.begin(),
              [fraction](float x) { return x * fraction; });
}

// Activation function that is used to introduce non-linearity in the model.
void relu(vector<float>& input) {
    transform(input.begin(), input.end(), input.begin(),
              [](float x) { return max(.0f, x); });
}

float crossEntropyLoss(vector<float>& output, float target) {
    // The cross entropy loss is a measure of how well the network is
    // performing. It is the difference between the predicted output and the
    // actual output.
    //
    // The cross entropy loss is calculated as follows:
    // -sum(y * log(y_hat))
    //
    // Where y is the actual output and y_hat is the predicted output.
    //
    // The cross entropy loss is used to update the weights and biases of the
    // network.
    return -log(output[target]);
}

float derivativeRelu(float& input) { return input > 0 ? 1.0f : .0f; }

// Forward pass
// The forward pass is the process of taking the input data and passing it
// through the neural network to get the output.
//
// First we calculate the dot product of the input and the weights, and add the
// bias. Then we apply the activation function to the result.
//
// We do this for each layer of the network.
float forward(Model& model, float target) {
    // 1. Calculate the dot product of the input layer and the weights, and add
    // the bias.
    // 2. Apply the activation function to the result (ReLU).
    // 3. Calculate the dot product of the hidden layer and the weights, and add
    // the bias
    // 4. Apply the activation function to the result (Softmax).

    for (size_t i = 0; i < model.hiddenLayer.size(); i++) {
        float sum = 0;
        for (size_t j = 0; j < model.inputLayer.size(); j++) {
            sum +=
                model.inputLayer[j] * model.w1[j + i * model.inputLayer.size()];
        }
        model.hiddenLayer[i] = sum + model.b1[i];
    }

    relu(model.hiddenLayer);

    for (size_t i = 0; i < model.outputLayer.size(); i++) {
        float sum = 0;
        for (size_t j = 0; j < model.hiddenLayer.size(); j++) {
            sum += model.hiddenLayer[j] *
                   model.w2[j + i * model.hiddenLayer.size()];
        }
        model.outputLayer[i] = sum + model.b2[i];
    }

    softmax(model.outputLayer);

    return crossEntropyLoss(model.outputLayer, target);
}

void updateWeights(Model& model, Gradients& gradients, Gradients& cache) {
    for (size_t i = 0; i < model.w2.size(); i++) {
        float grad = gradients.dW2[i];
        cache.dW2[i] =
            decay_rate * cache.dW2[i] + (1 - decay_rate) * (grad * grad);
        model.w2[i] -=
            model.learningRate * grad / (sqrt(cache.dW2[i]) + epsilon);
    }

    // Update b2 parameters
    for (size_t i = 0; i < model.b2.size(); i++) {
        float grad = gradients.dB2[i];
        cache.dB2[i] =
            decay_rate * cache.dB2[i] + (1 - decay_rate) * (grad * grad);
        model.b2[i] -=
            model.learningRate * grad / (sqrt(cache.dB2[i]) + epsilon);
    }

    // Update w1 parameters
    for (size_t i = 0; i < model.w1.size(); i++) {
        float grad = gradients.dW1[i];
        cache.dW1[i] =
            decay_rate * cache.dW1[i] + (1 - decay_rate) * (grad * grad);
        model.w1[i] -=
            model.learningRate * grad / (sqrt(cache.dW1[i]) + epsilon);
    }

    // Update b1 parameters
    for (size_t i = 0; i < model.b1.size(); i++) {
        float grad = gradients.dB1[i];
        cache.dB1[i] =
            decay_rate * cache.dB1[i] + (1 - decay_rate) * (grad * grad);
        model.b1[i] -=
            model.learningRate * grad / (sqrt(cache.dB1[i]) + epsilon);
    }
}

void backward(Model& model, float target, Gradients& gradients) {
    // This is the backward pass of the network. It is the process of updating
    // the weights and biases of the network based on the error of the output.
    // It uses backpropagation to calculate the gradients of the weights and
    // biases.
    //
    //
    // The network can essentially be seen as a function that takes the input
    // and transforms it into an output. The output is dependent on the weights,
    // biases, and activation functions of the network.
    //
    // dC/dW = dC/dA * dA/dZ * dZ/dW
    // dC/dB = dC/dA * dA/dZ * dZ/dB
    // dC/dA = dC/dZ * dZ/dA

    // This is the partial derivative of the cost w.r.t the weights
    for (size_t i = 0; i < model.outputLayer.size(); i++) {
        // Because we are using the softmax function, the derivative of the
        // output layer is simply the output layer itself minus 1 for the target
        // class.
        gradients.dOutput[i] +=
            (i == target) ? model.outputLayer[i] - 1 : model.outputLayer[i];

        for (size_t j = 0; j < model.hiddenLayer.size(); j++) {
            gradients.dW2[i * model.hiddenLayer.size() + j] +=
                gradients.dOutput[i] * model.hiddenLayer[j];
        }
    }

    // This is the partial derivative of the cost w.r.t the biases
    for (size_t i = 0; i < model.b2.size(); i++) {
        gradients.dB2[i] += gradients.dOutput[i];
    }

    // This is the partial derivative of the cost w.r.t the hidden layer
    // For backpropagation to work we need to align the shape of the weights
    // with the shape of the gradient. This is why we need to transpose the
    // weights.
    for (size_t i = 0; i < model.hiddenLayer.size(); i++) {
        auto sum = 0.0f;
        for (size_t j = 0; j < model.outputLayer.size(); j++) {
            sum += gradients.dOutput[j] *
                   model.w2[j * model.hiddenLayer.size() + i];
        }
        gradients.dHidden[i] += sum * derivativeRelu(model.hiddenLayer[i]);
    }

    for (size_t i = 0; i < model.hiddenLayer.size(); i++) {
        for (size_t j = 0; j < model.inputLayer.size(); j++) {
            gradients.dW1[i * model.inputLayer.size() + j] +=
                gradients.dHidden[i] * model.inputLayer[j];
        }
    }

    for (size_t i = 0; i < model.b1.size(); i++) {
        gradients.dB1[i] += gradients.dHidden[i];
    }
}

void averageGradients(Gradients& gradients, int batchSize) {
    auto averageRate = 1.0 / batchSize;
    transform(gradients.dW1.begin(), gradients.dW1.end(), gradients.dW1.begin(),
              [averageRate](float x) { return x * averageRate; });
    transform(gradients.dB1.begin(), gradients.dB1.end(), gradients.dB1.begin(),
              [averageRate](float x) { return x * averageRate; });
    transform(gradients.dW2.begin(), gradients.dW2.end(), gradients.dW2.begin(),
              [averageRate](float x) { return x * averageRate; });
    transform(gradients.dB2.begin(), gradients.dB2.end(), gradients.dB2.begin(),
              [averageRate](float x) { return x * averageRate; });
    transform(gradients.dHidden.begin(), gradients.dHidden.end(),
              gradients.dHidden.begin(),
              [averageRate](float x) { return x * averageRate; });
    transform(gradients.dOutput.begin(), gradients.dOutput.end(),
              gradients.dOutput.begin(),
              [averageRate](float x) { return x * averageRate; });
}

// For feed forward neural networks it's important to initialize the weights as
// small random numbers.
// This is because stochastic gradient descent is sensitive to the initial
// values of the weights.
void initializeWeights(vector<float>& weights, float n) {
    float stddev = sqrt(2.0 / n);
    normal_distribution<float> normal(0, stddev);
    random_device rd;
    default_random_engine generator(rd());
    transform(weights.begin(), weights.end(), weights.begin(),
              [&normal, &generator](auto var) {
                  (void)var;
                  return normal(generator);
              });
}

void reinitializeGradients(Gradients& gradients) {
    fill(gradients.dW1.begin(), gradients.dW1.end(), .0f);
    fill(gradients.dB1.begin(), gradients.dB1.end(), .0f);
    fill(gradients.dW2.begin(), gradients.dW2.end(), .0f);
    fill(gradients.dB2.begin(), gradients.dB2.end(), .0f);
    fill(gradients.dHidden.begin(), gradients.dHidden.end(), .0f);
    fill(gradients.dOutput.begin(), gradients.dOutput.end(), .0f);
}

float train(Model& model, Gradients& gradients, Gradients& cacheGradients,
            vector<float>& labels, vector<float>& images, int batchSize,
            int numBatches) {
    // -- Stochastic Gradient Descent --
    // We will use batches in power of 2, this is because it is more efficient
    // to use powers of 2 when working with SIMD instructions.
    //
    // We will run all examples in parallel, and then update the weights and
    // biases.
    //
    // 1. Spin up a thread for each example in a batch.
    // 2. Run the forward pass for each thread.
    // 3. Average the loss of each thread.
    // 4. Run the backward pass for the average loss.
    // 6. Repeat until convergence.
    //
    // Obvious optimizations:
    // 1. Use SIMD instructions.
    // 2. Use a worker pool to avoid the overhead of spinning up threads.
    //
    // But start with the simplest implementation first, and then optimize.
    //

    vector<int> indices(images.size() / 784);
    iota(indices.begin(), indices.end(), 0);
    random_device rd;
    mt19937 gen(rd());
    shuffle(indices.begin(), indices.end(), gen);

    auto totalLoss = .0f;
    for (int i = 0; i < numBatches; i++) {
        for (int j = 0; j < batchSize; j++) {
            auto idx = indices[i * batchSize + j];
            auto start = idx * 784;
            auto end = start + 784;
            auto input =
                vector<float>(images.begin() + start, images.begin() + end);
            model.inputLayer = input;
            totalLoss += forward(model, labels[idx]);
            backward(model, labels[idx], gradients);
        }
        averageGradients(gradients, batchSize);
        updateWeights(model, gradients, cacheGradients);
        reinitializeGradients(gradients);
    }
    return totalLoss / (numBatches * batchSize);
}

int predict(Model& model, vector<float>& image) {
    model.inputLayer = image;
    forward(model, 0);
    return distance(
        model.outputLayer.begin(),
        max_element(model.outputLayer.begin(), model.outputLayer.end()));
}

int main() {
    auto labelsPath = "mnist/train-labels.idx1-ubyte";
    auto imagesPath = "mnist/train-images.idx3-ubyte";
    auto labels = loadLabels(labelsPath);
    auto images = loadImages(imagesPath);

    auto testLabelsPath = "mnist/t10k-labels.idx1-ubyte";
    auto testImagesPath = "mnist/t10k-images.idx3-ubyte";

    auto testLabels = loadLabels(testLabelsPath);
    auto testImages = loadImages(testImagesPath);

    // Theoretically, since we are finding the correlatoin between the
    // pixels of
    // the image and the label, we need to map 28 * 28 pixels to a label,
    // which
    // is 784 pixels.
    //
    // Since we need to find a correlation between each pixel and the
    // label, we
    // need two layers which are fully connected.

    // -- Model Architecture --
    // 1. Input Layer:  784 neurons,        weights: 784 x 64,      biases: 64
    // 2. Hidden Layer: 512 neurons,        weights: 512 x 10,      biases: 10
    // 3. Output Layer: 10 neurons
    Model model{
        .learningRate = 0.001,
        .inputLayer = vector<float>(784),
        .hiddenLayer = vector<float>(512),
        .outputLayer = vector<float>(10),
        .w1 = vector<float>(784 * 512),
        .b1 = vector<float>(512),
        .w2 = vector<float>(512 * 10),
        .b2 = vector<float>(10),
    };

    // Images per batch
    auto batchSize = 32;
    // Multiply by 784 to get the number of pixels in the image.
    auto averageRate = 1.0 / (batchSize * 784);
    auto numBatches = images.size() * averageRate;
    auto epochs = 15;

    auto gradients = Gradients{
        .dW1 = vector<float>(model.w1.size(), .0f),
        .dB1 = vector<float>(model.b1.size(), .0f),
        .dW2 = vector<float>(model.w2.size(), .0f),
        .dB2 = vector<float>(model.b2.size(), .0f),
        .dHidden = vector<float>(model.hiddenLayer.size(), .0f),
        .dOutput = vector<float>(model.outputLayer.size(), .0f),
    };

    auto cacheGradients = Gradients{
        .dW1 = vector<float>(model.w1.size(), .0f),
        .dB1 = vector<float>(model.b1.size(), .0f),
        .dW2 = vector<float>(model.w2.size(), .0f),
        .dB2 = vector<float>(model.b2.size(), .0f),
        .dHidden = vector<float>(model.hiddenLayer.size(), .0f),
        .dOutput = vector<float>(model.outputLayer.size(), .0f),
    };

    initializeWeights(model.w1, model.inputLayer.size());
    initializeWeights(model.w2, model.hiddenLayer.size());
    initializeWeights(model.b1, model.hiddenLayer.size());
    initializeWeights(model.b2, model.outputLayer.size());

    auto t1 = benchmark_start();
    for (int i = 0; i <= epochs; i++) {
        auto epoch_loss = train(model, gradients, cacheGradients, labels,
                                images, batchSize, numBatches);
        cout << "Epoch: " << i << " Loss: " << epoch_loss << '\n';
    }
    benchmark_end(t1);

    auto correct = 0;
    auto numTestImages = testImages.size() / 784;
    for (size_t i = 0; i < numTestImages; i++) {
        auto idx = i * 784;
        auto image = vector<float>(testImages.begin() + idx,
                                   testImages.begin() + idx + 784);
        auto prediction = predict(model, image);
        // cout << "Prediction: " << prediction << " Actual: " << testLabels[i]
        //      << '\n';
        if (prediction == testLabels[i]) {
            correct++;
        }
    }
    cout << "Accuracy: " << (correct / static_cast<float>(numTestImages)) * 100
         << "%\n";

    return 0;
}
