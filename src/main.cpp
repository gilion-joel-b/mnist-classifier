#include <fstream>
#include <ios>
#include <iostream>
#include <vector>

std::vector<int> loadLabels(std::string path) {
    std::ifstream labels(path, std::ios_base::binary);
    std::vector<int> labelValues;

    int labelsLength = labels.tellg();

    if (labels) {
        labels.seekg(0, labels.end);

        int rows = labelsLength - 8;
        for (int i = 8; i < rows; i++) {
            char c;
            labels.seekg(i, labels.beg);
            labels.read(&c, 1);
            labelValues.push_back(c);
        }
    }

    return labelValues;
}

auto loadImages(std::string path) {
    std::ifstream images(path, std::ios_base::binary);
    std::vector<std::vector<int>> imageValues;

    if (images) {
        images.seekg(0, images.end);
        int imagesLength = images.tellg();
        images.seekg(16, images.beg);

        int rows = imagesLength - 16;

        for (int i = 16; i < rows; i++) {
            std::vector<int> image;

            for (int j = 0; j < 28; j++) {
                char c;
                images.seekg(i, images.beg);
                images.read(&c, 1);
                image.push_back(c);
            }
            imageValues.push_back(image);
        }

    }
}

int main() {
    std::ifstream images("mnist/train-labels.idx1-ubyte",
                         std::ios_base::binary);
    auto labels = loadLabels("mnist/train-labels.idx1-ubyte");
}
