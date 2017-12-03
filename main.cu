//19/10/17
//ALOUI Driss
//HPCA

//Source : https://www.cc.gatech.edu/~bader/papers/GPUMergePath-ICS2012.pdf

#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define NB 1
#define NTPB 4
#define N NTPB*NB

#define TAILLE 12

//Fct Merge
__host__ __device__ void merge(type* A, int na, int aid, type* B, int nb, int bid, type* C, int cid, int load) {

  for(int t=0; t<load; t++) {
    if(A[aid] < B[bid]) {
      C[cid+t] = A[aid];
      aid++;
    }else {
      C[cid+t] = B[bid];
      bid++;
    }
  }
}

//Fct GPU partitionning
__global__ void GPUpartitionning(type* A, int na, type* B, int nb, type* C){
  int a, b, offset;
  int aid = 0;
  int bid = 0;

  int tid = blockIdx.x*blockDim.x+threadIdx.x;// identifiant de thread
  int index = tid*(na+nb)/(blockDim.x * gridDim.x);// index de debut dans C
  int a_top = (index>na)? na:index;
  int b_top = (index>na)? index-na:0;
  int a_bot = b_top;

// binary search for diagonal intersectios
  while(true) {
    offset = (a_top - a_bot) / 2;
    a = a_top - offset;
    b = b_top + offset;

    if(A[a]>B[b-1]){
      if(A[a-1]<=B[b]){
        aid = a;
        bid = b;
        break; //point trouve !
      }else{
        a_top = a-1;
        b_top = b+1;
      }
    }else{
      a_bot = a+1;
    }
  }
// merge
  merge(A, na, aid, B, nb, bid, C, index, (na+nb)/(blockDim.x * gridDim.x));
}


//main
int main(){

  cudaEvent_t start, stop;
  cudaEventCreate ( &start );
  cudaEventCreate ( &stop );


//Alloc Array
  // cpu
  int n = TAILLE;
  int* cpu_v = (int*)malloc(n*sizeof(int));
  int* out = (int*)malloc(n*sizeof(int));

  // gpu
  int na = int(n/2);
  int nb = n-na;

  int *A, *B, *C;
  cudaMalloc(&A, na*sizeof(int));
  cudaMalloc(&B, nb*sizeof(int));
  cudaMalloc(&C, n*sizeof(int));

  if(!A || !B || !C ) {
    printf("memory alloc error\n");
    return -1;
  }

//init Array
  for(int i=0;i<na;i++){
    cpu_v[i]=i;
  }
  for(int i=na;i<n;i++){
    cpu_v[i]=i-na;
  }

//Cpu vers Gpu
  int error1 = cudaMemcpy(A, cpu_v, na*sizeof(int), cudaMemcpyHostToDevice);
  int error2 = cudaMemcpy(B, cpu_v+na, nb*sizeof(int), cudaMemcpyHostToDevice);
  printf("error1 %d (Cpu vers Gpu)\n",error1); // 0 donc bien
  printf("error2 %d (Cpu vers Gpu)\n",error2);

//partitionning
  cudaEventRecord(start);
  GPUpartitionning<<<NB,NTPB>>>(A, na, B, nb, C);
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

//Gpu vers cpu
  int error3 = cudaMemcpy(out, C, n*sizeof(int), cudaMemcpyDeviceToHost);
  printf("error3 %d (Gpu vers cpu)\n",error3);

//Time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("%f ms\n",milliseconds);

//free
  free(cpu_v);
  free(out);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

// return 0
  return 0;
}

