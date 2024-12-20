
#include <vector>

class Matrix
{
private:
  std::vector<std::vector<int> > matrix;
  Matrix (std::vector<std::vector<int> > matrix) { this->matrix = matrix; }
};
