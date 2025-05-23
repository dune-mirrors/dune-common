
#include <span>
#include <benchmark/benchmark.h>

#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>
#include <dune/common/tensor.hh>

static void BM_assignment_1d(benchmark::State& state) {
  Dune::Tensor<double, std::dynamic_extent> A(1000000);
  double value = 1.0;
  for (auto _ : state)
    A = value++;
}
BENCHMARK(BM_assignment_1d);

static void BM_assignment_2d(benchmark::State& state) {
  Dune::Tensor<double, std::dynamic_extent, std::dynamic_extent> A(1000,1000);
  double value = 1.0;
  for (auto _ : state)
    A = value++;
}
BENCHMARK(BM_assignment_2d);

static void BM_assignment_3d(benchmark::State& state) {
  Dune::Tensor<double, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent> A(100,100,100);
  double value = 1.0;
  for (auto _ : state)
    A = value++;
}
BENCHMARK(BM_assignment_3d);

static void BM_assignment_DenseVector(benchmark::State& state) {
  Dune::DynamicVector<double> A(1000000);
  double value = 1.0;
  for (auto _ : state)
    A = value++;
}
BENCHMARK(BM_assignment_DenseVector);

static void BM_assignment_DenseMatrix(benchmark::State& state) {
  Dune::DynamicMatrix<double> A(1000,1000);
  double value = 1.0;
  for (auto _ : state)
    A = value++;
}
BENCHMARK(BM_assignment_DenseMatrix);


static void BM_frobeniusnorm2_1d(benchmark::State& state) {
  Dune::Tensor<double, std::dynamic_extent> A(1000000);
  A = 0.01;
  double sum = 0.0;
  for (auto _ : state)
    benchmark::DoNotOptimize(sum += A.frobenius_norm2());
}
BENCHMARK(BM_frobeniusnorm2_1d);

static void BM_frobeniusnorm2_2d(benchmark::State& state) {
  Dune::Tensor<double, std::dynamic_extent, std::dynamic_extent> A(1000,1000);
  A = 0.01;
  double sum = 0.0;
  for (auto _ : state)
    benchmark::DoNotOptimize(sum += A.frobenius_norm2());
}
BENCHMARK(BM_frobeniusnorm2_2d);

static void BM_frobeniusnorm2_3d(benchmark::State& state) {
  Dune::Tensor<double, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent> A(100,100,100);
  A = 0.01;
  double sum = 0.0;
  for (auto _ : state)
    benchmark::DoNotOptimize(sum += A.frobenius_norm2());
}
BENCHMARK(BM_frobeniusnorm2_3d);

static void BM_frobeniusnorm2_DenseVector(benchmark::State& state) {
  Dune::DynamicVector<double> A(1000000);
  A = 0.01;
  double sum = 0.0;
  for (auto _ : state)
    benchmark::DoNotOptimize(sum += A.two_norm2());
}
BENCHMARK(BM_frobeniusnorm2_DenseVector);

static void BM_frobeniusnorm2_DenseMatrix(benchmark::State& state) {
  Dune::DynamicMatrix<double> A(1000,1000);
  A = 0.01;
  double sum = 0.0;
  for (auto _ : state)
    benchmark::DoNotOptimize(sum += A.frobenius_norm2());
}
BENCHMARK(BM_frobeniusnorm2_DenseMatrix);


BENCHMARK_MAIN();