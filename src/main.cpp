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

int main() {
    std::ifstream images("mnist/train-labels.idx1-ubyte",
                         std::ios_base::binary);
    auto labels = loadLabels("mnist/train-labels.idx1-ubyte");
}
