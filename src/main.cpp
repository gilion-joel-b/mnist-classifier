#include <fstream>
#include <vector>

class NeuralNetwork {
    // remember to transpose matrice before multiplication -- great for
    // performance
  public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    std::vector<float> forward(const std::vector<float>& input);
    void backward(const std::vector<float>& input,
                  const std::vector<float>& target);
    void update_weights(float learning_rate);

  private:
    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<std::vector<float>> weights_hidden_output;
    std::vector<float> hidden_layer;
    std::vector<float> output_layer;
};

int main() {
    std::ifstream labels("../mnist/train-labels.idx1-ubyte");
    std::ifstream images("../mnist/train-labels.idx1-ubyte");
}
