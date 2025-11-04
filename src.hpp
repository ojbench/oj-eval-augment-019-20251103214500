#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Build K_concat and V_concat by concatenating keys[0..i] and values[0..i]
    Matrix* K_concat = matrix_memory_allocator.Allocate("K_concat_" + std::to_string(i) + "_0");
    Matrix* V_concat = matrix_memory_allocator.Allocate("V_concat_" + std::to_string(i) + "_0");
    gpu_sim.Copy(keys[0], K_concat, kInGpuHbm);
    gpu_sim.Copy(values[0], V_concat, kInGpuHbm);

    for (size_t j = 1; j <= i; ++j) {
      Matrix* new_K_concat = matrix_memory_allocator.Allocate("K_concat_" + std::to_string(i) + "_" + std::to_string(j));
      Matrix* new_V_concat = matrix_memory_allocator.Allocate("V_concat_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.Concat(K_concat, keys[j], new_K_concat, 0, kInGpuHbm);
      gpu_sim.Concat(V_concat, values[j], new_V_concat, 0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(K_concat);
      gpu_sim.ReleaseMatrix(V_concat);
      K_concat = new_K_concat;
      V_concat = new_V_concat;
    }

    // Move matrices to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_concat);

    // Transpose K_concat
    gpu_sim.Transpose(K_concat, kInSharedMemory);

    // Compute Q * K^T
    Matrix* QK = matrix_memory_allocator.Allocate("QK_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_concat, QK);
    gpu_sim.ReleaseMatrix(K_concat);

    // Compute exp(QK)
    Matrix* exp_QK = matrix_memory_allocator.Allocate("exp_QK_" + std::to_string(i));
    gpu_sim.MatExp(QK, exp_QK);
    gpu_sim.ReleaseMatrix(QK);

    // Compute softmax row by row
    Matrix* softmax_QK = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.GetRow(exp_QK, row, exp_row, kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.Sum(exp_row, row_sum);

      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.MatDiv(exp_row, row_sum, softmax_row);

      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);

      if (softmax_QK == nullptr) {
        softmax_QK = softmax_row;
      } else {
        Matrix* new_softmax_QK = matrix_memory_allocator.Allocate("softmax_QK_new_" + std::to_string(i) + "_" + std::to_string(row));
        gpu_sim.Concat(softmax_QK, softmax_row, new_softmax_QK, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_QK);
        gpu_sim.ReleaseMatrix(softmax_row);
        softmax_QK = new_softmax_QK;
      }
    }
    gpu_sim.ReleaseMatrix(exp_QK);

    // Move V_concat to SRAM
    gpu_sim.MoveMatrixToSharedMem(V_concat);

    // Compute softmax_QK * V_concat
    Matrix* result = matrix_memory_allocator.Allocate("result_" + std::to_string(i));
    gpu_sim.MatMul(softmax_QK, V_concat, result);

    gpu_sim.ReleaseMatrix(softmax_QK);
    gpu_sim.ReleaseMatrix(V_concat);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    // Run the simulator
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer
    rater.CommitAnswer(*result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu