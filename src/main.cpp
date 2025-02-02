#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

void benchmark(string ref, string arg, function<vector<int>(string)> f) {
    using chrono::duration;
    using chrono::duration_cast;
    using chrono::high_resolution_clock;
    using chrono::milliseconds;

    cout << "Benchmarking " << ref << endl;

    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        auto labels = f(arg);
    }

    auto t2 = high_resolution_clock::now();
    // Getting number of milliseconds as an integer.
    auto ms_int = duration_cast<chrono::milliseconds>(t2 - t1);

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

    int imageSize = 28 * 28;
    int numImages = (imagesLength - 16);

    vector<unsigned char> buffer(imageSize * numImages);
    images.read(reinterpret_cast<char*>(buffer.data()), numImages);

    // Instead of returning a list of 28x28 matrices, we can return a list
    // of all values.
    // This is because this way we can store it in contiguous memory which
    // improves cache locality, which is important for performance.
    // Also, it allows us to use SIMD instructions to process the data.
    vector<int> imageValues(imageSize);
    for (int i = 0; i < numImages; i++) {
        imageValues[i] = static_cast<int>(buffer[i]);
    }

    return imageValues;
}

// Activation function that is used to squash the output of the network to a
// probability distribution.
void softmax(vector<float>& input) {
    auto sum = accumulate(input.begin(), input.end(), 0.0,
                          [](float a, float b) { return a + exp(b); });

    auto fraction = 1.0 / sum;
    transform(input.begin(), input.end(), input.begin(),
              [sum, fraction](float x) { return exp(x) * fraction; });
}

float derivativeSoftmax(float input) { return 0; }

// Activation function that is used to introduce non-linearity in the model.
void relu(vector<float>& input) {
    transform(input.begin(), input.end(), input.begin(),
              [](int x) { return max(0, x); });
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
void forward(vector<int>& inputLayer, vector<float>& w1, vector<float>& b1,
             vector<float>& hiddenLayer, vector<float>& w2, vector<float>& b2,
             vector<float>& outputLayer) {
    // 1. Calculate the dot product of the input layer and the weights, and add
    // the bias.
    // 2. Apply the activation function to the result (ReLU).
    // 3. Calculate the dot product of the hidden layer and the weights, and add
    // the bias
    // 4. Apply the activation function to the result (Softmax).

    for (int j = 0; j < 64; j++) {
        float sum = 0;
        for (int i = 0; i < 784; i++) {
            sum += inputLayer[i] * w1[i + j * 784];
        }
        hiddenLayer[j] = sum + b1[j];
    }

    relu(hiddenLayer);

    for (int j = 0; j < 10; j++) {
        float sum = 0;
        for (int i = 0; i < 64; i++) {
            sum += hiddenLayer[i] * w2[i + j * 64];
        }
        outputLayer[j] = sum + b2[j];
    }

    softmax(outputLayer);
}

void backward(vector<int>& inputLayer, vector<float>& w1, vector<float>& b1,
              vector<float>& hiddenLayer, vector<float>& w2, vector<float>& b2,
              vector<float>& outputLayer, int target) {
    float learningRate = 0.01;
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
    vector<float> gradientOutput(outputLayer.size());
    for (int i = 0; i < outputLayer.size(); i++) {
        // Because we are using the softmax function, the derivative of the
        // output layer is simply the output layer itself minus 1 for the target
        // class.
        gradientOutput[i] = (i == target) ? outputLayer[i] - 1 : outputLayer[i];

        for (int j = 0; j < hiddenLayer.size(); j++) {
            w2[j * outputLayer.size() + i] -=
                learningRate * gradientOutput[i] * hiddenLayer[j];
        }
    }

    // This is the partial derivative of the cost w.r.t the biases
    for (int i = 0; i < b2.size(); i++) {
        b2[i] -= learningRate * gradientOutput[i];
    }

    // This is the partial derivative of the cost w.r.t the hidden layer
    // For backpropagation to work we need to align the shape of the weights
    // with the shape of the gradient. This is why we need to transpose the
    // weights.
    vector<float> gradientHidden(hiddenLayer.size(), .0f);
    for (int i = 0; i < hiddenLayer.size(); i++) {
        for (int j = 0; j < outputLayer.size(); j++) {
            gradientHidden[i] +=
                gradientOutput[j] * w2[i * outputLayer.size() + j];
        }
        gradientHidden[i] *= derivativeRelu(hiddenLayer[i]);
    }

    // We update the weights of the hidden layer.
    for (int i = 0; i < hiddenLayer.size(); i++) {
        for (int j = 0; j < inputLayer.size(); j++) {
            w1[j * hiddenLayer.size() + i] -=
                learningRate * gradientHidden[i] * inputLayer[j];
        }
    }

    // finally we update the biases of the hidden layer.
    for (int i = 0; i < b1.size(); i++) {
        b1[i] -= learningRate * gradientHidden[i];
    }
}

// For feed forward neural networks it's important to initialize the weights as
// small random numbers.
// This is because stochastic gradient descent is sensitive to the initial
// values of the weights.
void initializeWeights(vector<float>& weights, int size) {
    auto fraction = 1.0 / RAND_MAX;
    transform(
        weights.begin(), weights.end(), weights.begin(),
        [fraction](float x) { return static_cast<float>(rand()) * fraction; });
}

void SGD(vector<int>& inputLayer, vector<float>& w1, vector<float>& b1,
         vector<float>& hiddenLayer, vector<float>& w2, vector<float>& b2,
         vector<float>& outputLayer, vector<int>& labels, vector<int>& images) {
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

    auto batchSize = 32;
    auto numBatches = images.size() / batchSize;
    auto averageOutput = vector<float>(outputLayer.size(), .0f);
    auto averageRate = 1.0 / batchSize;

    for (int i = 0; i < numBatches; i++) {
        for (int j = 0; j < batchSize; j++) {
            auto start = j * 784;
            auto end = start + 784;
            auto input =
                vector<int>(images.begin() + start, images.begin() + end);
            forward(input, w1, b1, hiddenLayer, w2, b2, outputLayer);
            transform(outputLayer.begin(), outputLayer.end(),
                      averageOutput.begin(), averageOutput.begin(),
                      plus<float>());
        }
        transform(averageOutput.begin(), averageOutput.end(),
                  averageOutput.begin(),
                  [averageRate](float x) { return x * averageRate; });
        backward(inputLayer, w1, b1, hiddenLayer, w2, b2, averageOutput,
                 labels[i]);
    }
}

int main() {
    auto labelsPath = "mnist/train-labels.idx1-ubyte";
    auto imagesPath = "mnist/train-images.idx3-ubyte";
    auto labels = loadLabels(labelsPath);
    auto images = loadImages(imagesPath);

    // Theoretically, since we are finding the correlatoin between the pixels of
    // the image and the label, we need to map 28 * 28 pixels to a label, which
    // is 784 pixels.
    //
    // Since we need to find a correlation between each pixel and the label, we
    // need two layers which are fully connected.

    // -- Model Architecture --
    // 1. Input Layer:  784 neurons,    weights: 784 x 64,  biases: 64
    // 2. Hidden Layer: 64 neurons,     weights: 64 x 10,   biases: 10
    // 3. Output Layer: 10 neurons

    auto inputLayer = vector<int>(784);
    auto w1 = vector<float>(784 * 64);
    auto b1 = vector<float>(64);

    auto hiddenLayer = vector<float>(64);
    auto w2 = vector<float>(64 * 10);
    auto b2 = vector<float>(10);

    auto outputLayer = vector<float>(10);

    return 0;
}
