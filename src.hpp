#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t num_elements = i + 1;

    // 1. Build K_mat [num_elements, d] and V_mat [num_elements, d]
    // Optimization: Instead of repeated concatenation in a loop, 
    // we can build them more efficiently.
    Matrix* K_mat = nullptr;
    Matrix* V_mat = nullptr;

    for (size_t j = 0; j < num_elements; ++j) {
      if (j == 0) {
        K_mat = matrix_memory_allocator.Allocate("K_0");
        V_mat = matrix_memory_allocator.Allocate("V_0");
        gpu_sim.Copy(keys[j], K_mat, Position::kInGpuHbm);
        gpu_sim.Copy(values[j], V_mat, Position::kInGpuHbm);
      } else {
        Matrix* next_K = matrix_memory_allocator.Allocate("K_acc_" + std::to_string(j));
        Matrix* next_V = matrix_memory_allocator.Allocate("V_acc_" + std::to_string(j));
        gpu_sim.Concat(K_mat, keys[j], next_K, 0, Position::kInGpuHbm);
        gpu_sim.Concat(V_mat, values[j], next_V, 0, Position::kInGpuHbm);
        gpu_sim.ReleaseMatrix(K_mat);
        gpu_sim.ReleaseMatrix(V_mat);
        K_mat = next_K;
        V_mat = next_V;
      }
    }
    
    // Transpose K to get K^T [d, num_elements]
    gpu_sim.Transpose(K_mat, Position::kInGpuHbm);

    // 2. Compute S = Q * K^T
    Matrix* S = matrix_memory_allocator.Allocate("S");
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_mat);
    gpu_sim.MatMul(current_query, K_mat, S);

    // 3. Softmax(S) row-wise
    Matrix* exp_S = matrix_memory_allocator.Allocate("exp_S");
    gpu_sim.MatExp(S, exp_S);

    Matrix* softmax_S = nullptr;
    for (size_t r = 0; r < num_elements; ++r) {
      Matrix* row = matrix_memory_allocator.Allocate("row_" + std::to_string(r));
      gpu_sim.GetRow(exp_S, r, row, Position::kInSharedMemory);
      
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(r));
      gpu_sim.Sum(row, row_sum);
      
      Matrix* row_softmax = matrix_memory_allocator.Allocate("row_softmax_" + std::to_string(r));
      gpu_sim.MatDiv(row, row_sum, row_softmax);
      
      if (r == 0) {
        softmax_S = matrix_memory_allocator.Allocate("s_acc_0");
        gpu_sim.Copy(row_softmax, softmax_S, Position::kInSharedMemory);
      } else {
        Matrix* next_s = matrix_memory_allocator.Allocate("s_acc_" + std::to_string(r));
        gpu_sim.Concat(softmax_S, row_softmax, next_s, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_S);
        softmax_S = next_s;
      }
      
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(row_softmax);
    }

    // 4. Result = softmax_S * V
    Matrix* Result = matrix_memory_allocator.Allocate("Result");
    gpu_sim.MoveMatrixToSharedMem(V_mat);
    gpu_sim.MatMul(softmax_S, V_mat, Result);
    gpu_sim.MoveMatrixToGpuHbm(Result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*Result);

    // Cleanup
    gpu_sim.ReleaseMatrix(K_mat);
    gpu_sim.ReleaseMatrix(V_mat);
    gpu_sim.ReleaseMatrix(S);
    gpu_sim.ReleaseMatrix(exp_S);
    gpu_sim.ReleaseMatrix(softmax_S);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
