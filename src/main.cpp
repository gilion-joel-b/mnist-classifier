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

// Activation function that is used to introduce non-linearity in the model.
void relu(vector<float>& input) {
    transform(input.begin(), input.end(), input.begin(),
              [](int x) { return max(0, x); });
}

// Loss function -- Mean Squared Error (MSE)
// This often leads to poor result as the gradients become very small.
// Cross-entropy loss is often used instead.
int mse(vector<float>& predicted, int target) {

    int sum = 0;
    for (int i = 0; i < predicted.size(); i++) {
        float val = predicted[i] * i;
        sum += (val - target) * (val - target);
    }

    return sum / predicted.size();
}

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

void backPropagation(vector<int>& inputLayer, vector<float>& w1,
                     vector<float>& b1, vector<float>& hiddenLayer,
                     vector<float>& w2, vector<float>& b2,
                     vector<float>& outputLayer, int target) {
    auto loss = mse(outputLayer, target);

    // The output layer is defined by the equation a = softmax(weights *
    // hiddenLayer + b2) The derivative of this function is easily derived
    // analytically.
    //
    // a = softmax(weights * hiddenLayer + b2) can be seens a a composition of
    // functions f . g . h, where f(x) = softmax(x), g(x) = x + b2, h(x) =
    // weights * hiddenLayer.
    //
    // This is the chain rule in calculus.

}

// Backward pass
// In the backward pass, we calculate the gradients of the loss function with
// respect to the weights and biases of the network.
//
// We use these gradients to update the weights and biases of the network using
// the gradient descent algorithm.
void backward(vector<int>& inputLayer, vector<float>& w1, vector<float>& b1,
              vector<float>& hiddenLayer, vector<float>& w2, vector<float>& b2,
              vector<float>& outputLayer, int target) {
    // 1, Calculate the gradients.
    // 2. Update the weights and biases of the network using gradient descent

    auto loss = mse(outputLayer, target);
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

    auto hiddenLayer = vector<float>(784);
    auto w2 = vector<float>(64 * 10);
    auto b2 = vector<float>(10);

    auto outputLayer = vector<float>(10);

    return 0;
}
