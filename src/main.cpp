#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<int> loadLabelsOne(std::string path) {
    std::ifstream labels(path, std::ios_base::binary);
    if (!labels) {
        throw std::runtime_error("Could not open file");
    }

    labels.seekg(0, labels.end);
    int labelsLength = labels.tellg();
    labels.seekg(0, labels.beg);

    int rows = labelsLength - 8;
    char buffer[labelsLength];
    std::vector<int> labelValues(rows);

    labels.read(buffer, labelsLength);

    for (int i = 8; i < labelsLength; i++) {
        labelValues[i - 8] = buffer[i];
    }

    return labelValues;
}

std::vector<int> loadLabelsThree(const std::string& path) {
    std::ifstream labels(path, std::ios_base::binary);
    if (!labels) {
        throw std::runtime_error("Could not open file");
    }

    labels.seekg(0, labels.end);
    int labelsLength = labels.tellg();
    int rows = labelsLength - 8;

    labels.seekg(8, labels.beg);

    std::vector<unsigned char> buffer(rows);
    labels.read(reinterpret_cast<char*>(buffer.data()), rows);

    std::vector<int> labelValues(rows);
    for (int i = 0; i < rows; ++i) {
        labelValues[i] = static_cast<int>(buffer[i]);
    }

    return labelValues;
}

void benchmark(std::string ref, std::string arg,
               std::function<std::vector<int>(std::string)> f) {
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    std::cout << "Benchmarking " << ref << std::endl;

    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        auto labels = f(arg);
    }

    auto t2 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<std::chrono::milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
}

int main() {
    auto path = "mnist/train-labels.idx1-ubyte";
    benchmark("one", path, &loadLabelsOne);
    benchmark("three", path, &loadLabelsThree);

    auto l1 = loadLabelsOne(path);
    auto l2 = loadLabelsThree(path);

    // std::cout << "l1: " << l1.size() << " l2:" << l2.size()
    //           << " l3:" << l3.size() << std::endl;

    // for (int i = 0; i < l1.size(); i++) {
    //     std::cout << l1[i] << " " << l2[i] << " " << l3[i] << std::endl;
    // }

    assert(l1.size() != 0);
    assert(l2.size() != 0);

    assert(l1 == l2);

    return 0;
}

// auto loadImages(std::string path) {
//     std::ifstream images(path, std::ios_base::binary);
//
//     if (!images) {
//         throw std::runtime_error("Could not open file");
//     }
//
//     // Need to offset by 16 to skip the header and rows x cols //
//     // After offsetting we know that each tensor is 28x28 matrix of pixels
//     //
//     // We actually want to return a list of 28x28 matrices
//     // First we need to iterate the the rows, and collect 28 at a time into
//     // a matrix.
//     //
//     // All values are unsigned bytes, in a list of 784 values ordered by
//     // rows.
//     //
//     // We can use a buffer to read all the values at once and then iterate
//     // through the buffer to create the meatrices.
//     //
//     // However this approach is not memory effieicent, to avoid this we can
//     // initialize empty vectors, and then stream the values per image which
//     is
//     // 28x28 = 784 values.
//     //
//     // This is much more memory efficient, and better for large datasets
//
//     images.seekg(0, images.end);
//     int imagesLength = images.tellg();
//     images.seekg(16, images.beg);
//
//     int imageSize = 28 * 28;
//     int numImages = (imagesLength - 16) / imageSize;
//
//     std::vector<std::vector<std::vector<int>>> imagesValues(
//         numImages, std::vector<std::vector<int>>(28, std::vector<int>(28)));
//
//     std::vector<unsigned char> buffer(imageSize * numImages);
//     std::vector<unsigned char> imageBuffer(imageSize);
// }
//
