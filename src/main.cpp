#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
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

// Activation function that is used to introduce non-linearity in the model.
void relu(vector<int>& input) {
    std::transform(input.begin(), input.end(), input.begin(),
                   [](int x) { return std::max(0, x); });
}

// Loss function -- Mean Squared Error (MSE)
//
int mse(vector<int>& predicted, vector<int>& actual) {
    assert(predicted.size() == actual.size());

    int sum = 0;
    for (int i = 0; i < predicted.size(); i++) {
        sum += (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
    }

    return sum / predicted.size();
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

    auto inputLayer = std::vector<int>(784);
    auto w1 = std::vector<int>(784 * 64);
    auto b1 = std::vector<int>(64);

    auto hiddenLayer = std::vector<int>(784);
    auto w2 = std::vector<int>(64 * 10);
    auto b2 = std::vector<int>(10);

    auto outputLayer = std::vector<int>(10);

    return 0;
}
