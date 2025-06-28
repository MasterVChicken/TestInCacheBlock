#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

#include "mgard/mgard-x/CompressionLowLevel/HybridHierarchyCompressor.hpp"
#include "mgard/mgard-x/Hierarchy/Hierarchy.hpp"
#include "mgard/mgard-x/RuntimeX/DataStructures/Array.hpp"
#include "mgard/mgard-x/RuntimeX/RuntimeX.h"

using namespace mgard_x;

template <DIM D, typename T>
void run(const std::vector<SIZE>& shape, const std::string& file_path, double eb) {
  using DeviceType = mgard_x::CUDA;
  SIZE N = 1;
  for (auto s : shape) N *= s;

  std::vector<T> data(N);
  std::ifstream fin(file_path, std::ios::binary);
  if (!fin) {
    std::cerr << "Failed to open file.\n";
    exit(1);
  }
  fin.read(reinterpret_cast<char *>(data.data()), N * sizeof(T));
  fin.close();

  Config config;
  config.num_local_refactoring_level = 1;
  config.lossless = lossless_type::CPU_Lossless;
  config.compress_with_dryrun = false;
  config.dev_type = mgard_x::device_type::CUDA;

  Hierarchy<D, T, DeviceType> hierarchy(shape, config);
  HybridHierarchyCompressor<D, T, DeviceType> compressor(hierarchy, config);
  compressor.Adapt(hierarchy, config, 0);

  T s = std::numeric_limits<double>::infinity(), norm;
  Array<D, T, DeviceType> original(shape, data.data());
  Array<1, Byte, DeviceType> compressed({N});
  Array<D, T, DeviceType> decompressed(shape);

  compressor.Compress(original, error_bound_type::REL, eb, s, norm, compressed, 0);
  compressor.Decompress(compressed, error_bound_type::REL, eb, s, norm, decompressed, 0);

  T *recon = decompressed.hostCopy();
  T max_error = 0;
  for (SIZE i = 0; i < N; ++i) {
    max_error = std::max(max_error, std::abs(data[i] - recon[i]));
  }

  std::cout << "L-inf norm: " << norm << std::endl;
  std::cout << "L-inf error: " << max_error << std::endl;
  std::cout << "Setting Error Bound: " << eb << std::endl;
  std::cout << "Actual Error Bound: " << max_error/norm << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <file_path> <precision=float/double> <dim=1~4> <dim_1> ... <dim_n> <eb>\n";
    return 1;
  }

  std::string file_path = argv[1];
  std::string precision = argv[2];
  int dim = std::stoi(argv[3]);

  if (dim < 1 || dim > 4 || argc < 4 + dim + 1) {
    std::cerr << "Invalid dimension or not enough shape arguments.\n";
    return 1;
  }

  std::vector<SIZE> shape(dim);
  for (int i = 0; i < dim; ++i) {
    shape[i] = std::stoul(argv[4 + i]);
  }

  double eb = std::stod(argv[4 + dim]);

  std::cout << "Reading file: " << file_path << "\n";

  if (precision == "float") {
    switch (dim) {
      case 1: run<1, float>(shape, file_path, eb); break;
      case 2: run<2, float>(shape, file_path, eb); break;
      case 3: run<3, float>(shape, file_path, eb); break;
      case 4: run<4, float>(shape, file_path, eb); break;
      default: std::cerr << "Unsupported dimension.\n"; return 1;
    }
  } else if (precision == "double") {
    switch (dim) {
      case 1: run<1, double>(shape, file_path, eb); break;
      case 2: run<2, double>(shape, file_path, eb); break;
      case 3: run<3, double>(shape, file_path, eb); break;
      case 4: run<4, double>(shape, file_path, eb); break;
      default: std::cerr << "Unsupported dimension.\n"; return 1;
    }
  } else {
    std::cerr << "Unsupported precision type: " << precision << "\n";
    return 1;
  }

  return 0;
}

