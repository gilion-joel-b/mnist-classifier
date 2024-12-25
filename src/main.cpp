#include <fstream>
#include <iostream>

int main() {
    std::ifstream labels("mnist/train-labels.idx1-ubyte");
    std::ifstream images("mnist/train-labels.idx1-ubyte");
    std::string line;

    if (labels) {
        labels.seekg(0, labels.end);
        int length = labels.tellg();
        int rows = length - 8;
        std::cout << "Total count: " << rows << std::endl;
        for (int i = 8; i < rows; i++) {
            char c;
            labels.seekg(i, labels.beg);
            labels.read(&c, 1);
            std::cout << "Label: " << (int)c << std::endl;
        }
    }

    return 0;
}
