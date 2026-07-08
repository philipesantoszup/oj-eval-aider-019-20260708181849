#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  for (size_t i = 0; i < keys.size(); ++i) {
    // Round i+1: we use keys[0...i] and values[0...i]
    // Q is provided by rater, shape [i+1, d]
    auto current_query = rater.GetNextQuery();
    size_t num_elements = i + 1;
    size_t d = keys[0]->GetColumnNum();

    // 1. Prepare K and V matrices for the current round
    // We need K as a matrix [d, num_elements] and V as [num_elements, d]
    // Since we can't easily "stack" in the simulator without Concat, 
    // and Concat is expensive, we'll build them.
    
    Matrix* K_mat = matrix_memory_allocator.Allocate("K_mat");
    Matrix* V_mat = matrix_memory_allocator.Allocate("V_mat");

    // Build V_mat [num_elements, d] by concatenating V_0...V_i
    Matrix* current_v_acc = nullptr;
    for (size_t j = 0; j < num_elements; ++j) {
      if (j == 0) {
        current_v_acc = matrix_memory_allocator.Allocate("v_acc_0");
        gpu_sim.Copy(values[j], current_v_acc, Position::kInGpuHbm);
      } else {
        Matrix* next_v_acc = matrix_memory_allocator.Allocate("v_acc_" + std::to_string(j));
        gpu_sim.Concat(current_v_acc, values[j], next_v_acc, 0, Position::kInGpuHbm);
        gpu_sim.ReleaseMatrix(current_v_acc);
        current_v_acc = next_v_acc;
      }
    }
    gpu_sim.Copy(current_v_acc, V_mat, Position::kInGpuHbm);
    gpu_sim.ReleaseMatrix(current_v_acc);

    // Build K_mat [num_elements, d] then transpose to [d, num_elements]
    Matrix* current_k_acc = nullptr;
    for (size_t j = 0; j < num_elements; ++j) {
      if (j == 0) {
        current_k_acc = matrix_memory_allocator.Allocate("k_acc_0");
        gpu_sim.Copy(keys[j], current_k_acc, Position::kInGpuHbm);
      } else {
        Matrix* next_k_acc = matrix_memory_allocator.Allocate("k_acc_" + std::to_string(j));
        gpu_sim.Concat(current_k_acc, keys[j], next_k_acc, 0, Position::kInGpuHbm);
        gpu_sim.ReleaseMatrix(current_k_acc);
        current_k_acc = next_k_acc;
      }
    }
    gpu_sim.Copy(current_k_acc, K_mat, Position::kInGpuHbm);
    gpu_sim.ReleaseMatrix(current_k_acc);
    
    // Transpose K to get K^T [d, num_elements]
    gpu_sim.Transpose(K_mat, Position::kInGpuHbm);

    // 2. Compute S = Q * K^T
    // Q: [num_elements, d], K^T: [d, num_elements] -> S: [num_elements, num_elements]
    Matrix* S = matrix_memory_allocator.Allocate("S");
    
    // Move to SRAM for calculation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_mat);
    
    // MatMul requires SRAM
    gpu_sim.MatMul(current_query, K_mat, S);

    // 3. Softmax(S) row-wise
    // Softmax(x) = exp(x) / sum(exp(x))
    Matrix* exp_S = matrix_memory_allocator.Allocate("exp_S");
    gpu_sim.MatExp(S, exp_S);

    // We need the sum of each row of exp_S. 
    // The simulator's Sum() returns a 1x1 matrix of the WHOLE matrix.
    // To do row-wise softmax, we must process row by row.
    Matrix* softmax_S = matrix_memory_allocator.Allocate("softmax_S");
    
    // Since we need to build softmax_S [num_elements, num_elements]
    Matrix* softmax_rows_acc = nullptr;
    for (size_t r = 0; r < num_elements; ++r) {
      Matrix* row = matrix_memory_allocator.Allocate("row_" + std::to_string(r));
      gpu_sim.GetRow(exp_S, r, row, Position::kInSharedMemory);
      
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(r));
      gpu_sim.Sum(row, row_sum);
      
      Matrix* row_softmax = matrix_memory_allocator.Allocate("row_softmax_" + std::to_string(r));
      gpu_sim.MatDiv(row, row_sum, row_softmax);
      
      if (r == 0) {
        softmax_rows_acc = matrix_memory_allocator.Allocate("s_acc_0");
        gpu_sim.Copy(row_softmax, softmax_rows_acc, Position::kInSharedMemory);
      } else {
        Matrix* next_s_acc = matrix_memory_allocator.Allocate("s_acc_" + std::to_string(r));
        gpu_sim.Concat(softmax_rows_acc, row_softmax, next_s_acc, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_rows_acc);
        softmax_rows_acc = next_s_acc;
      }
      
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(row_softmax);
    }
    gpu_sim.Copy(softmax_rows_acc, softmax_S, Position::kInSharedMemory);
    gpu_sim.ReleaseMatrix(softmax_rows_acc);

    // 4. Result = softmax_S * V
    // softmax_S: [num_elements, num_elements], V: [num_elements, d] -> Result: [num_elements, d]
    Matrix* Result = matrix_memory_allocator.Allocate("Result");
    gpu_sim.MoveMatrixToSharedMem(V_mat);
    gpu_sim.MatMul(softmax_S, V_mat, Result);
    
    // Move result to HBM for committing
    gpu_sim.MoveMatrixToGpuHbm(Result);

    // Run simulator to execute the queue
    gpu_sim.Run(false, &matrix_memory_allocator);
    
    // Commit the answer
    rater.CommitAnswer(*Result);

    // Cleanup temporary matrices that aren't managed by the allocator's automatic release 
    // (though the prompt says allocator manages them, we should be explicit if needed, 
    // but the simulator's ReleaseMatrix is the way to free GPU memory).
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
