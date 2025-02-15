#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <numeric>
#include <ratio>
#include <stdexcept>
#include <string>
#include <vector>

using std::vector, std::transform, std::accumulate, std::fill, std::max,
    std::ifstream, std::runtime_error, std::string, std::cout, std::endl,
    std::ios_base, std::function;

void benchmark(string ref, string arg, function<vector<int>(string)> f) {
    using std::milli, std::chrono::duration, std::chrono::duration_cast,
        std::chrono::high_resolution_clock, std::chrono::milliseconds;

    cout << "Benchmarking " << ref << endl;

    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        auto labels = f(arg);
    }

    auto t2 = high_resolution_clock::now();
    // Getting number of milliseconds as an integer.
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    // Getting number of milliseconds as a double.
    duration<double, milli> ms_double = t2 - t1;

    cout << ms_int.count() << "ms\n";
    cout << ms_double.count() << "ms\n";
}

vector<int> loadLabels(const string& path) {
    ifstream labels(path, ios_base::binary);
    if (!labels) {
        throw runtime_error("Could not open file");
    }

    labels.seekg(0, labels.end);
    int labelsLength = labels.tellg();
    int rows = labelsLength - 8;

    labels.seekg(8, labels.beg);

    vector<unsigned char> buffer(rows);
    labels.read(reinterpret_cast<char*>(buffer.data()), rows);

    vector<int> labelValues(rows);
    for (int i = 0; i < rows; ++i) {
        labelValues[i] = static_cast<int>(buffer[i]);
    }

    return labelValues;
}

auto loadImages(string path) {
    ifstream images(path, ios_base::binary);

    if (!images) {
        throw runtime_error("Could not open file");
    }

    // Need to offset by 16 to skip the header and rows x cols //
    // After offsetting we know that each tensor is 28x28 matrix of pixels
    //
    // We actually want to return a list of 28x28 matrices
    // First we need to iterate the the rows, and collect 28 at a time into
    // a matrix.
    //
    // All values are unsigned bytes, in a list of 784 values ordered by
    // rows.
    //
    // We can use a buffer to read all the values at once and then iterate
    // through the buffer to create the meatrices.
    //
    images.seekg(0, images.end);
    int imagesLength = images.tellg();
    images.seekg(16, images.beg);

    int pixels = (imagesLength - 16);

    vector<unsigned char> buffer(pixels);
    images.read(reinterpret_cast<char*>(buffer.data()), pixels);

    // Instead of returning a list of 28x28 matrices, we can return a list
    // of all values.
    // This is because this way we can store it in contiguous memory which
    // improves cache locality, which is important for performance.
    // Also, it allows us to use SIMD instructions to process the data.
    vector<float> imageValues(pixels);
    for (int i = 0; i < pixels; i++) {
        imageValues[i] = static_cast<float>(buffer[i]);
    }

    return imageValues;
}

struct Model {
    float learningRate;
    vector<int> inputLayer;
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
void softmax(vector<float>& input) {
    auto sum = accumulate(input.begin(), input.end(), 0.0,
                          [](float a, float b) { return a + exp(b); });

    auto fraction = 1.0 / sum;
    transform(input.begin(), input.end(), input.begin(),
              [sum, fraction](float x) { return exp(x) * fraction; });
}

// Activation function that is used to introduce non-linearity in the model.
void relu(vector<float>& input) {
    transform(input.begin(), input.end(), input.begin(),
              [](float x) { return max(.0f, x); });
}

int crossEntropyLoss(vector<float>& output, int target) {
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
float forward(Model& model, int target) {
    // 1. Calculate the dot product of the input layer and the weights, and add
    // the bias.
    // 2. Apply the activation function to the result (ReLU).
    // 3. Calculate the dot product of the hidden layer and the weights, and add
    // the bias
    // 4. Apply the activation function to the result (Softmax).

    for (int j = 0; j < 64; j++) {
        float sum = 0;
        for (int i = 0; i < 784; i++) {
            sum += model.inputLayer[i] * model.w1[i + j * 784];
        }
        model.hiddenLayer[j] = sum + model.b1[j];
    }

    relu(model.hiddenLayer);

    for (int j = 0; j < 10; j++) {
        float sum = 0;
        for (int i = 0; i < 64; i++) {
            sum += model.hiddenLayer[i] * model.w2[i + j * 64];
        }
        model.outputLayer[j] = sum + model.b2[j];
    }

    softmax(model.outputLayer);

    return crossEntropyLoss(model.outputLayer, target);
}

void updateWeights(Model& model, Gradients& gradients) {
    for (int i = 0; i < model.outputLayer.size(); i++) {
        for (int j = 0; j < model.hiddenLayer.size(); j++) {
            model.w2[i * model.hiddenLayer.size() + j] -= model.learningRate *
                                                          gradients.dOutput[i] *
                                                          model.hiddenLayer[j];
        }
    }

    for (int i = 0; i < model.b2.size(); i++) {
        model.b2[i] -= model.learningRate * gradients.dOutput[i];
    }

    for (int i = 0; i < model.hiddenLayer.size(); i++) {
        for (int j = 0; j < model.inputLayer.size(); j++) {
            model.w1[i * model.inputLayer.size() + j] -=
                model.learningRate * gradients.dHidden[i] * model.inputLayer[j];
        }
    }

    for (int i = 0; i < model.b1.size(); i++) {
        model.b1[i] -= model.learningRate * gradients.dHidden[i];
    }
}

void backward(Model& model, int target, Gradients& gradients) {
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
    for (int i = 0; i < model.outputLayer.size(); i++) {
        // Because we are using the softmax function, the derivative of the
        // output layer is simply the output layer itself minus 1 for the target
        // class.
        gradients.dOutput[i] +=
            (i == target) ? model.outputLayer[i] - 1 : model.outputLayer[i];

        for (int j = 0; j < model.hiddenLayer.size(); j++) {
            gradients.dW2[i * model.hiddenLayer.size() + j] =
                gradients.dOutput[i] * model.hiddenLayer[j];
        }
    }

    // This is the partial derivative of the cost w.r.t the biases
    for (int i = 0; i < model.b2.size(); i++) {
        gradients.dB2[i] += gradients.dOutput[i];
    }

    // This is the partial derivative of the cost w.r.t the hidden layer
    // For backpropagation to work we need to align the shape of the weights
    // with the shape of the gradient. This is why we need to transpose the
    // weights.
    for (int i = 0; i < model.hiddenLayer.size(); i++) {
        auto sum = 0.0f;
        for (int j = 0; j < model.outputLayer.size(); j++) {
            sum += gradients.dOutput[j] *
                   model.w2[i * model.outputLayer.size() + j];
        }
        gradients.dHidden[i] += sum * derivativeRelu(model.hiddenLayer[i]);
    }

    for (int i = 0; i < model.hiddenLayer.size(); i++) {
        for (int j = 0; j < model.inputLayer.size(); j++) {
            gradients.dW1[i * model.hiddenLayer.size() + j] =
                gradients.dHidden[i] * model.inputLayer[j];
        }
    }

    for (int i = 0; i < model.b1.size(); i++) {
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
void initializeWeights(vector<float>& weights) {
    auto fraction = 1.0 / RAND_MAX;
    transform(weights.begin(), weights.end(), weights.begin(),
              [fraction](float x) {
                  return static_cast<float>(rand()) * fraction - 0.5f;
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

float train(Model& model, Gradients& gradients, vector<int>& labels,
            vector<float>& images, int batchSize, int numBatches) {
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

    auto totalLoss = .0f;
    for (int i = 0; i < numBatches; i++) {
        for (int j = 0; j < batchSize; j++) {
            auto idx = i * batchSize + j;
            auto start = idx * 784;
            auto end = start + 784;
            auto input =
                vector<int>(images.begin() + start, images.begin() + end);
            model.inputLayer = input;
            totalLoss += forward(model, labels[idx]);
            backward(model, labels[idx], gradients);
        }
        averageGradients(gradients, batchSize);
        updateWeights(model, gradients);
        reinitializeGradients(gradients);
    }
    return totalLoss / (numBatches * batchSize);
}

int predict(Model& model, vector<int>& image) {
    model.inputLayer = image;
    forward(model, 0);
    return std::distance(
        model.outputLayer.begin(),
        std::max_element(model.outputLayer.begin(), model.outputLayer.end()));
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
    // 1. Input Layer:  784 neurons,    weights: 784 x 64,  biases: 64
    // 2. Hidden Layer: 64 neurons,     weights: 64 x 10,   biases: 10
    // 3. Output Layer: 10 neurons
    Model model{
        .learningRate = 0.01,
        .inputLayer = vector<int>(784),
        .hiddenLayer = vector<float>(64),
        .outputLayer = vector<float>(10),
        .w1 = vector<float>(784 * 64),
        .b1 = vector<float>(64),
        .w2 = vector<float>(64 * 10),
        .b2 = vector<float>(10),
    };

    // Images per batch
    auto batchSize = 32;
    // Multiply by 784 to get the number of pixels in the image.
    auto averageRate = 1.0 / (batchSize * 784);
    auto numBatches = images.size() * averageRate;
    auto epochs = 10;

    auto gradients = Gradients{
        .dW1 = vector<float>(model.w1.size(), .0f),
        .dB1 = vector<float>(model.b1.size(), .0f),
        .dW2 = vector<float>(model.w2.size(), .0f),
        .dB2 = vector<float>(model.b2.size(), .0f),
        .dHidden = vector<float>(model.hiddenLayer.size(), .0f),
        .dOutput = vector<float>(model.outputLayer.size(), .0f),
    };

    initializeWeights(model.w1);
    initializeWeights(model.w2);

    for (int i = 0; i < epochs; i++) {
        auto epoch_loss =
            train(model, gradients, labels, images, batchSize, numBatches);
        cout << "Epoch: " << i << " Loss: " << epoch_loss << '\n';
    }

    for (int i = 0; i < 10; i++) {
        auto idx = i * 784;
        auto image = vector<int>(testImages.begin() + idx,
                                 testImages.begin() + idx + 784);
        auto prediction = predict(model, image);
        cout << "Prediction: " << prediction << " Actual: " << testLabels[i]
             << '\n';
    }

    return 0;
}
