#include <cstdlib>
#include <fstream>
#include <iostream>

auto main() -> int {
  const std::string name{"size-options.txt"};
  std::ofstream out(name);
  for (int i{0}; i < 3 * 1e3; ++i) {
    if (i % 64 == 0 && i % 10 == 2) {
      out << "consider using " << i;
      out << ", which is " << std::div(i, 64).quot;
      out << " * 64, for your Nx and Ny values" << std::endl;
    }
  }
  return 0;
}
