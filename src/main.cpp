#include <cstdio>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<int> loadLabels(std::string path) {
    std::ifstream labels(path, std::ios_base::binary);
    if (!labels) {
        throw std::runtime_error("Could not open file");
    }

    int labelsLength = labels.tellg();
    std::vector<int> labelValues(labelsLength);

    labels.seekg(0, labels.end);

    int rows = labelsLength - 8;
    for (int i = 8; i < rows; i++) {
        char c;
        labels.seekg(i, labels.beg);
        labels.read(&c, 1);
        labelValues.push_back(c);
    }

    return labelValues;
}

int main() {
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    std::ifstream images("mnist/train-labels.idx1-ubyte",
                         std::ios_base::binary);
    auto t1 = high_resolution_clock::now();

    auto labels = loadLabels("mnist/train-labels.idx1-ubyte");
    auto t2 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

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
