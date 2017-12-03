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
__host__ __device__ void merge(int* A, int na, int aid, int* B, int nb, int bid, int* C, int cid, int load) {

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
__global__ void GPUpartitionning(int* A, int na, int* B, int nb, int* C){
  int a, b, offset;
  int aid;
  int bid;

  int tid = blockIdx.x*blockDim.x+threadIdx.x;// identifiant de thread
  int index = tid*(na+nb)/(blockDim.x * gridDim.x);// index de debut dans C
  int a_top = (index>na)? na:index;
  int b_top = (index>na)? index-na:0;
  int a_bot = b_top;
if(tid ==0) {
  aid = 0;
  bid = 0;
}else{
// binary search for diagonal intersectios
  while(true) {
printf("offset %d a_top %d a_bot %d a %d b %d aid %d bid %d\n",offset,a_top,a_bot,a,b,aid,bid);
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
}
// merge
printf("merge\n");
printf("na %d aid %d nb %d bid %d index %d (na+nb)/(blockDim.x * gridDim.x) %d \n", na, aid, nb, bid, index, (na+nb)/(blockDim.x * gridDim.x));
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
  int* T_in = (int*)malloc(n*sizeof(int));
  int* T_out = (int*)malloc(n*sizeof(int));

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
printf("\n A : \n");
  for(int i=0;i<na;i++){
    T_in[i]=i;
printf("%d\t",T_in[i]);
  }
printf("\n B : \n");
  for(int i=na;i<n;i++){
    T_in[i]=i-na;
printf("%d\t",T_in[i]);
  }
printf("\n");

//Cpu vers Gpu
  int error1 = cudaMemcpy(A, T_in, na*sizeof(int), cudaMemcpyHostToDevice);
  int error2 = cudaMemcpy(B, T_in+na, nb*sizeof(int), cudaMemcpyHostToDevice);
  printf("error1 %d (Cpu vers Gpu)\n",error1); // 0 donc bien
  printf("error2 %d (Cpu vers Gpu)\n",error2);

//partitionning
  cudaEventRecord(start);
printf("GPUpartitionning NB %d NTPB %d na %d nb %d\n",NB,NTPB,na,nb);
  GPUpartitionning<<<NB,NTPB>>>(A, na, B, nb, C);
  cudaEventRecord(stop);
printf("cudaDeviceSynchronize\n");
  cudaDeviceSynchronize();

//Gpu vers cpu
  int error3 = cudaMemcpy(T_out, C, n*sizeof(int), cudaMemcpyDeviceToHost);
  printf("error3 %d (Gpu vers cpu)\n",error3);

//Time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("%f ms\n",milliseconds);

printf("\n T_in : \n");
  for(int i=0;i<n;i++){
printf("%d\t",T_in[i]);
  }

printf("\n T_out %d : \n",n);
  for(int i=0;i<n-1;i++){
printf("%d\t",T_out[i]);
  }

printf("\n T_out %d : \n",n);
  for(int i=0;i<n;i++){
printf("%d\t",T_out[i]);
  }
printf("\nT_out[11] %d\n",T_out[11]);
//free
  free(T_in);
  free(T_out);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

// return 0
  return 0;
}

