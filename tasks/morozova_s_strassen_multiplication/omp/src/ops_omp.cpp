#include "morozova_s_strassen_multiplication/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "morozova_s_strassen_multiplication/common/include/common.hpp"

namespace morozova_s_strassen_multiplication {

namespace {

Matrix AddMatrixImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(static) shared(a, b, result, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result(i, j) = a(i, j) + b(i, j);
    }
  }

  return result;
}

Matrix SubtractMatrixImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(static) shared(a, b, result, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result(i, j) = a(i, j) - b(i, j);
    }
  }

  return result;
}

Matrix MultiplyStandardImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(dynamic, 1) shared(a, b, result, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += a(i, k) * b(k, j);
      }
      result(i, j) = sum;
    }
  }

  return result;
}

void SplitMatrixImpl(const Matrix &m, Matrix &m11, Matrix &m12, Matrix &m21, Matrix &m22) {
  int n = m.size;
  int half = n / 2;

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      m11(i, j) = m(i, j);
      m12(i, j) = m(i, j + half);
      m21(i, j) = m(i + half, j);
      m22(i, j) = m(i + half, j + half);
    }
  }
}

Matrix MergeMatricesImpl(const Matrix &m11, const Matrix &m12, const Matrix &m21, const Matrix &m22) {
  int half = m11.size;
  int n = 2 * half;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(static) shared(m11, m12, m21, m22, result, half)
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      result(i, j) = m11(i, j);
      result(i, j + half) = m12(i, j);
      result(i + half, j) = m21(i, j);
      result(i + half, j + half) = m22(i, j);
    }
  }

  return result;
}

Matrix MultiplyStandardParallelImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel default(none) shared(a, b, result, n)
  {
#pragma omp for collapse(2) schedule(dynamic, 1)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
          sum += a(i, k) * b(k, j);
        }
        result(i, j) = sum;
      }
    }
  }

  return result;
}

struct StrassenTempMatrices {
  Matrix a11, a12, a21, a22;
  Matrix b11, b12, b21, b22;
};

void ComputeStrassenLevel(const Matrix &a, const Matrix &b, Matrix &result, int leaf_size, int max_depth,
                          int current_depth) {
  int n = a.size;

  if (n <= leaf_size || n % 2 != 0) {
    result = MultiplyStandardParallelImpl(a, b);
    return;
  }

  int half = n / 2;

  Matrix a11(half), a12(half), a21(half), a22(half);
  Matrix b11(half), b12(half), b21(half), b22(half);
  Matrix c11(half), c12(half), c21(half), c22(half);

  SplitMatrixImpl(a, a11, a12, a21, a22);
  SplitMatrixImpl(b, b11, b12, b21, b22);

  std::vector<Matrix> p(7, Matrix(half));

  if (current_depth < max_depth) {
#pragma omp parallel sections default(none) \
    shared(a11, a12, a21, a22, b11, b12, b21, b22, leaf_size, max_depth, current_depth, p, half)
    {
#pragma omp section
      {
        Matrix temp = SubtractMatrixImpl(b12, b22);
        Matrix temp_result;
        ComputeStrassenLevel(a11, temp, temp_result, leaf_size, max_depth, current_depth + 1);
        p[0] = std::move(temp_result);
      }

#pragma omp section
      {
        Matrix temp = AddMatrixImpl(a11, a12);
        Matrix temp_result;
        ComputeStrassenLevel(temp, b22, temp_result, leaf_size, max_depth, current_depth + 1);
        p[1] = std::move(temp_result);
      }

#pragma omp section
      {
        Matrix temp = AddMatrixImpl(a21, a22);
        Matrix temp_result;
        ComputeStrassenLevel(temp, b11, temp_result, leaf_size, max_depth, current_depth + 1);
        p[2] = std::move(temp_result);
      }

#pragma omp section
      {
        Matrix temp = SubtractMatrixImpl(b21, b11);
        Matrix temp_result;
        ComputeStrassenLevel(a22, temp, temp_result, leaf_size, max_depth, current_depth + 1);
        p[3] = std::move(temp_result);
      }

#pragma omp section
      {
        Matrix temp1 = AddMatrixImpl(a11, a22);
        Matrix temp2 = AddMatrixImpl(b11, b22);
        Matrix temp_result;
        ComputeStrassenLevel(temp1, temp2, temp_result, leaf_size, max_depth, current_depth + 1);
        p[4] = std::move(temp_result);
      }

#pragma omp section
      {
        Matrix temp1 = SubtractMatrixImpl(a12, a22);
        Matrix temp2 = AddMatrixImpl(b21, b22);
        Matrix temp_result;
        ComputeStrassenLevel(temp1, temp2, temp_result, leaf_size, max_depth, current_depth + 1);
        p[5] = std::move(temp_result);
      }

#pragma omp section
      {
        Matrix temp1 = SubtractMatrixImpl(a11, a21);
        Matrix temp2 = AddMatrixImpl(b11, b12);
        Matrix temp_result;
        ComputeStrassenLevel(temp1, temp2, temp_result, leaf_size, max_depth, current_depth + 1);
        p[6] = std::move(temp_result);
      }
    }
  } else {
    Matrix temp1, temp2, temp_result;

    temp1 = SubtractMatrixImpl(b12, b22);
    ComputeStrassenLevel(a11, temp1, temp_result, leaf_size, max_depth, current_depth + 1);
    p[0] = std::move(temp_result);

    temp1 = AddMatrixImpl(a11, a12);
    ComputeStrassenLevel(temp1, b22, temp_result, leaf_size, max_depth, current_depth + 1);
    p[1] = std::move(temp_result);

    temp1 = AddMatrixImpl(a21, a22);
    ComputeStrassenLevel(temp1, b11, temp_result, leaf_size, max_depth, current_depth + 1);
    p[2] = std::move(temp_result);

    temp1 = SubtractMatrixImpl(b21, b11);
    ComputeStrassenLevel(a22, temp1, temp_result, leaf_size, max_depth, current_depth + 1);
    p[3] = std::move(temp_result);

    temp1 = AddMatrixImpl(a11, a22);
    temp2 = AddMatrixImpl(b11, b22);
    ComputeStrassenLevel(temp1, temp2, temp_result, leaf_size, max_depth, current_depth + 1);
    p[4] = std::move(temp_result);

    temp1 = SubtractMatrixImpl(a12, a22);
    temp2 = AddMatrixImpl(b21, b22);
    ComputeStrassenLevel(temp1, temp2, temp_result, leaf_size, max_depth, current_depth + 1);
    p[5] = std::move(temp_result);

    temp1 = SubtractMatrixImpl(a11, a21);
    temp2 = AddMatrixImpl(b11, b12);
    ComputeStrassenLevel(temp1, temp2, temp_result, leaf_size, max_depth, current_depth + 1);
    p[6] = std::move(temp_result);
  }

  Matrix temp;

  temp = AddMatrixImpl(p[4], p[3]);
  temp = SubtractMatrixImpl(temp, p[1]);
  c11 = AddMatrixImpl(temp, p[5]);

  c12 = AddMatrixImpl(p[0], p[1]);
  c21 = AddMatrixImpl(p[2], p[3]);

  temp = AddMatrixImpl(p[4], p[0]);
  temp = SubtractMatrixImpl(temp, p[2]);
  c22 = SubtractMatrixImpl(temp, p[6]);

  result = MergeMatricesImpl(c11, c12, c21, c22);
}

}  // namespace

MorozovaSStrassenMultiplicationOMP::MorozovaSStrassenMultiplicationOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool MorozovaSStrassenMultiplicationOMP::ValidationImpl() {
  return true;
}

bool MorozovaSStrassenMultiplicationOMP::PreProcessingImpl() {
  if (GetInput().empty()) {
    valid_data_ = false;
    return true;
  }

  double size_val = GetInput()[0];
  if (size_val <= 0.0) {
    valid_data_ = false;
    return true;
  }

  int n = static_cast<int>(size_val);

  if (GetInput().size() != 1 + (2 * static_cast<size_t>(n) * static_cast<size_t>(n))) {
    valid_data_ = false;
    return true;
  }

  valid_data_ = true;
  n_ = n;

  a_ = Matrix(n_);
  b_ = Matrix(n_);

  int idx = 1;
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      a_(i, j) = GetInput()[idx++];
    }
  }

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      b_(i, j) = GetInput()[idx++];
    }
  }

  return true;
}

bool MorozovaSStrassenMultiplicationOMP::RunImpl() {
  if (!valid_data_) {
    return true;
  }

  const int leaf_size = 64;

  if (n_ <= leaf_size) {
    c_ = MultiplyStandardParallelImpl(a_, b_);
  } else {
    ComputeStrassenLevel(a_, b_, c_, leaf_size, kMaxParallelDepth, 0);
  }

  return true;
}

bool MorozovaSStrassenMultiplicationOMP::PostProcessingImpl() {
  OutType &output = GetOutput();
  output.clear();

  if (!valid_data_) {
    return true;
  }

  output.push_back(static_cast<double>(n_));

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      output.push_back(c_(i, j));
    }
  }

  return true;
}

Matrix MorozovaSStrassenMultiplicationOMP::AddMatrix(const Matrix &a, const Matrix &b) {
  return AddMatrixImpl(a, b);
}

Matrix MorozovaSStrassenMultiplicationOMP::SubtractMatrix(const Matrix &a, const Matrix &b) {
  return SubtractMatrixImpl(a, b);
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStandard(const Matrix &a, const Matrix &b) {
  return MultiplyStandardImpl(a, b);
}

void MorozovaSStrassenMultiplicationOMP::SplitMatrix(const Matrix &m, Matrix &m11, Matrix &m12, Matrix &m21,
                                                     Matrix &m22) {
  SplitMatrixImpl(m, m11, m12, m21, m22);
}

Matrix MorozovaSStrassenMultiplicationOMP::MergeMatrices(const Matrix &m11, const Matrix &m12, const Matrix &m21,
                                                         const Matrix &m22) {
  return MergeMatricesImpl(m11, m12, m21, m22);
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStrassen(const Matrix &a, const Matrix &b, int leaf_size) {
  Matrix result;
  ComputeStrassenLevel(a, b, result, leaf_size, kMaxParallelDepth, 0);
  return result;
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStandardParallel(const Matrix &a, const Matrix &b) {
  return MultiplyStandardParallelImpl(a, b);
}

}  // namespace morozova_s_strassen_multiplication
