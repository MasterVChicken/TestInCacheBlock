#include <cmath>
#include <iostream>
#include <vector>

#include "mgard/mgard-x/RuntimeX/RuntimeX.h"
#include "mgard/mgard-x/RuntimeX/DataStructures/SubArray.hpp"
#include "mgard/mgard-x/DataRefactoring/HybridHierarchyDataRefactor.hpp"

using namespace mgard_x;

static constexpr DIM D = 3;
using T = float;
using DeviceType = SERIAL;

int main() {
  // std::vector<SIZE> shape = {64, 64, 64};
  std::vector<SIZE> shape = {512, 512, 512};
  const SIZE N = shape[0] * shape[1] * shape[2];

  std::vector<T> v(N), v_recon(N);

  // for (SIZE i = 0; i < N; i++) {
  //   SIZE x = i % shape[2];
  //   SIZE y = (i / shape[2]) % shape[1];
  //   SIZE z = i / (shape[2] * shape[1]);
  //   v[i] = x + 2 * y + 4 * z;
  // }

  const char *filename =
      "/home/leonli/SDRBENCH/single_precision/SDRBENCH-EXASKY-NYX-512x512x512/"
      "temperature.f32";
  std::ifstream fin(filename, std::ios::binary);
  if (!fin) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return 1;
  }
  fin.read(reinterpret_cast<char *>(v.data()), N * sizeof(T));
  if (!fin) {
    std::cerr << "Failed to read data from file: " << filename << std::endl;
    return 1;
  }
  fin.close();
  std::cout << "Successfully read data from file: " << filename << std::endl;

  // We only perform local data refactor from 8x8x8 -> 5x5x5
  Config config;
  config.num_local_refactoring_level = 1;  //

  Hierarchy<D, T, DeviceType> hierarchy(shape, config);
  data_refactoring::HybridHierarchyDataRefactor<D, T, DeviceType> hybrid(
      hierarchy, config);

  const SIZE coeff_size = hybrid.DecomposedDataSize();
  std::vector<T> coeff_buffer(coeff_size);
  SubArray<1, T, DeviceType> coeff_subarray({coeff_size}, coeff_buffer.data());

  SubArray<D, T, DeviceType> sub_in(shape, v.data());
  SubArray<D, T, DeviceType> sub_out(shape, v_recon.data());

  // Decompose + Recompose 
  hybrid.Decompose(sub_in, coeff_subarray, /*queue_idx=*/0);
  hybrid.Recompose(sub_out, coeff_subarray, /*queue_idx=*/0);

  // Error check
  T max_error = 0;
  for (SIZE i = 0; i < N; i++) {
    max_error = std::max(max_error, std::abs(v[i] - v_recon[i]));
  }
  std::cout << "Max absolute error = " << max_error << std::endl;

  return 0;
}
