
#include <cstddef>
#include <vector>

class Tensor {
  private:
    std::vector<std::vector<int>> tensor;
    Tensor(std::vector<std::vector<int>> tensor) { this->tensor = tensor; }

  public:
    int shape() { return tensor.size(); }
};
