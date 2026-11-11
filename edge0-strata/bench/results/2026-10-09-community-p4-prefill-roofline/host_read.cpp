#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <omp.h>
int main(){
 const size_t bytes=2ull<<30,n=bytes/sizeof(uint64_t);uint64_t* p=(uint64_t*)std::aligned_alloc(64,bytes);
 if(!p)return 1;omp_set_num_threads(28);
 #pragma omp parallel for
 for(size_t i=0;i<n;++i)p[i]=i&255;
 for(int trial=0;trial<3;++trial){
  uint64_t sum=0;auto start=std::chrono::steady_clock::now();
  for(int r=0;r<8;++r){
   #pragma omp parallel for reduction(+:sum)
   for(size_t i=0;i<n;++i)sum+=p[i];
  }
  double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  printf("{\"kind\":\"host_read\",\"threads\":28,\"trial\":%d,\"GB_s\":%.6f,\"checksum\":%llu}\n",trial,bytes*8/s/1e9,(unsigned long long)sum);
 }std::free(p);
}
