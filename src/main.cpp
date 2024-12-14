#include <vector>

class NeuralNetwork {
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
    std::vector<std::vector<float>> load_mnist_images(
        const std::string& filepath);
    std::vector<int> load_mnist_labels(const std::string& filepath);

    return 0;
}
