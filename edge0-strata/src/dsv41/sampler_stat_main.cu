// CUDA standalone of the sampler statistics harness: nvcc -include ds4_launch.h -I include -I src/dsv41 sampler_stat_main.cu
#include "sampler_stat.inc"
#include "vendor/sampler.cu"
int main(int argc, char** argv) { return sampler_stat::run(argc, argv); }
