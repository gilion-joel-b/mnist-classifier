#include <fstream>
#include <ios>
#include <string>
#include <vector>

using std::vector, std::string, std::ifstream, std::runtime_error,
    std::ios_base;

inline vector<float> loadLabels(const string& path) {
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

    vector<float> labelValues(rows);
    for (int i = 0; i < rows; ++i) {
        labelValues[i] = static_cast<float>(buffer[i]);
    }

    return labelValues;
}

inline auto loadImages(const string& path) {
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
        imageValues[i] = static_cast<float>(buffer[i]) / 255.0f;
    }

    return imageValues;
}
