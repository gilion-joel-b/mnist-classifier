#include <fstream>
#include <ios>
#include <iostream>

int main() {
    std::ifstream labels("mnist/train-labels.idx1-ubyte",
                         std::ios_base::binary);
    std::ifstream images("mnist/train-labels.idx1-ubyte",
                         std::ios_base::binary);

    int labelsLength = labels.tellg();
    int labelValues[labelsLength];

    if (labels) {
        labels.seekg(0, labels.end);

        int rows = labelsLength - 8;
        std::cout << "Total count: " << rows << std::endl;
        for (int i = 8; i < rows; i++) {
            char c;
            labels.seekg(i, labels.beg);
            labels.read(&c, 1);
            labelValues[i] = c;
        }
        std::cout << "Row count: " << rows << std::endl;
    }

    return 0;
}

int* loadLabels(std::string path) {
    std::ifstream labels(path, std::ios_base::binary);

    int labelsLength = labels.tellg();
    int labelValues[labelsLength];

    if (labels) {
        labels.seekg(0, labels.end);

        int rows = labelsLength - 8;
        for (int i = 8; i < rows; i++) {
            char c;
            labels.seekg(i, labels.beg);
            labels.read(&c, 1);
            labelValues[i] = c;
        }
    }

    return labelValues;
}
