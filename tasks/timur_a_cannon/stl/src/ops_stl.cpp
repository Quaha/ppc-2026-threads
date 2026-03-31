#include "timur_a_cannon/stl/include/ops_stl.hpp"

#include <cmath>
#include <thread>
#include <vector>

#include "util/include/util.hpp"

namespace timur_a_cannon {

TimurACannonMatrixMultiplicationSTL::TimurACannonMatrixMultiplicationSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TimurACannonMatrixMultiplicationSTL::ValidationImpl() {
  const auto &input = GetInput();
  int b_size = std::get<0>(input);
  const auto &mat_a = std::get<1>(input);
  const auto &mat_b = std::get<2>(input);

  if (b_size <= 0 || mat_a.empty() || mat_b.empty()) {
    return false;
  }
  size_t n = mat_a.size();
  return (n == mat_a[0].size() && n == mat_b.size() && n == mat_b[0].size() && (n % b_size == 0));
}

bool TimurACannonMatrixMultiplicationSTL::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void TimurACannonMatrixMultiplicationSTL::BlockMultiplyAccumulate(const std::vector<std::vector<double>> &a,
                                                                  const std::vector<std::vector<double>> &b,
                                                                  std::vector<std::vector<double>> &c, int b_size) {
  for (int i = 0; i < b_size; ++i) {
    for (int k = 0; k < b_size; ++k) {
      double temp = a[i][k];
      for (int j = 0; j < b_size; ++j) {
        c[i][j] += temp * b[k][j];
      }
    }
  }
}

bool TimurACannonMatrixMultiplicationSTL::RunImpl() {
  const auto &input = GetInput();
  int b_size = std::get<0>(input);
  const auto &src_a = std::get<1>(input);
  const auto &src_b = std::get<2>(input);
  int n = static_cast<int>(src_a.size());
  int grid_sz = n / b_size;

  using Matrix = std::vector<std::vector<double>>;
  using BlockGrid = std::vector<std::vector<Matrix>>;

  BlockGrid bl_a(grid_sz, std::vector<Matrix>(grid_sz, Matrix(b_size, std::vector<double>(b_size))));
  BlockGrid bl_b(grid_sz, std::vector<Matrix>(grid_sz, Matrix(b_size, std::vector<double>(b_size))));
  BlockGrid bl_c(grid_sz, std::vector<Matrix>(grid_sz, Matrix(b_size, std::vector<double>(b_size, 0.0))));

  int num_threads = ppc::util::GetNumThreads();

  auto parallel_grid_op = [&](auto func) {
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t]() {
        for (int i = t; i < grid_sz; i += num_threads) {
          for (int j = 0; j < grid_sz; ++j) {
            func(i, j);
          }
        }
      });
    }
    for (auto &th : threads) {
      th.join();
    }
  };

  parallel_grid_op([&](int i, int j) {
    int shift = (i + j) % grid_sz;
    for (int r = 0; r < b_size; ++r) {
      for (int c = 0; c < b_size; ++c) {
        bl_a[i][j][r][c] = src_a[i * b_size + r][shift * b_size + c];
        bl_b[i][j][r][c] = src_b[shift * b_size + r][j * b_size + c];
      }
    }
  });

  for (int step = 0; step < grid_sz; ++step) {
    parallel_grid_op([&](int i, int j) { BlockMultiplyAccumulate(bl_a[i][j], bl_b[i][j], bl_c[i][j], b_size); });

    if (step < grid_sz - 1) {
      std::vector<std::thread> shift_threads;
      shift_threads.emplace_back([&]() {
        for (int i = 0; i < grid_sz; ++i) {
          auto first = std::move(bl_a[i][0]);
          for (int j = 0; j < grid_sz - 1; ++j) {
            bl_a[i][j] = std::move(bl_a[i][j + 1]);
          }
          bl_a[i][grid_sz - 1] = std::move(first);
        }
      });
      shift_threads.emplace_back([&]() {
        for (int j = 0; j < grid_sz; ++j) {
          auto first = std::move(bl_b[0][j]);
          for (int i = 0; i < grid_sz - 1; ++i) {
            bl_b[i][j] = std::move(bl_b[i + 1][j]);
          }
          bl_b[grid_sz - 1][j] = std::move(first);
        }
      });
      for (auto &th : shift_threads) {
        th.join();
      }
    }
  }

  Matrix res_mat(n, std::vector<double>(n));
  parallel_grid_op([&](int i, int j) {
    for (int r = 0; r < b_size; ++r) {
      for (int c = 0; c < b_size; ++c) {
        res_mat[i * b_size + r][j * b_size + c] = bl_c[i][j][r][c];
      }
    }
  });

  GetOutput() = std::move(res_mat);
  return true;
}

bool TimurACannonMatrixMultiplicationSTL::PostProcessingImpl() {
  return true;
}

}  // namespace timur_a_cannon
