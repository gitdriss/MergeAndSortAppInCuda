//19/10/17
//ALOUI Driss
//DO Alexandre
//HPCA

//Source : https://www.cc.gatech.edu/~bader/papers/GPUMergePath-ICS2012.pdf

#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

#define NB 1
#define NTPB 4
#define N NTPB*NB

#define TAILLE 12

//Fct Merge
__host__ __device__ void merge(int* A, int na, int aid, int* B, int nb, int bid, int* C, int cid, int T) {

  for(int i=0; i<T; i++) {
    if(aid<na && bid<nb) {
      if(A[aid] < B[bid]) {
        C[cid+i] = A[aid];
        aid++;
      }else {
        C[cid+i] = B[bid];
        bid++;
      }
    }else{ // Pour derniers indices (reste plus q un tableau)
      if(aid<na){
        C[cid+i] = A[aid];
        aid++;
      }else{
        C[cid+i] = B[bid];
        bid++;
      }
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
printf("[%d] Call merge\n",tid);
  merge(A, na, aid, B, nb, bid, C, index, (na+nb)/(blockDim.x * gridDim.x));
}

//intro
void intro(){
//INTRODUCTION
  system("clear");
  printf("Dans le cadre\n");
  printf("\n");
  sleep(1);
  printf("N9-IPA PARALLELISME AVANCE HPCA\n");
  printf("Projet de fin de module\n");
  printf("\n");
  sleep(1);
  printf("2017 - 2018\n");
  sleep(2);

  system("clear");
  printf("Supervisé par\n");
  printf("\n");
  printf("Lokman ABBAS TURKI\n");
  sleep(2);

  system("clear");
  printf("ALOUI Driss\n");
  printf("DO Alexandre\n");
  printf("\n");
  sleep(1);
  printf("MAIN 5 Polytech Paris UPMC\n");
  sleep(2);

  system("clear");
  printf("Présente\n");
  sleep(1);

  system("clear");
  printf("MergeAndSortAppInCuda\n");
  sleep(1);

  system("clear");
  char s;
  printf("Press enter to continue\n");
  s=getchar();
  putchar(s);
  system("clear");
}

//main
int main(){
intro();

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
printf("\n init Array A and B\n");
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
  if(error1)
    printf("error1 %d (Cpu vers Gpu)\n",error1);
  if(error2)
    printf("error2 %d (Cpu vers Gpu)\n",error2);

//partitionning
  cudaEventRecord(start);
printf("\nCall GPUpartitionning NB %d NTPB %d na %d nb %d\n",NB,NTPB,na,nb);
  GPUpartitionning<<<NB,NTPB>>>(A, na, B, nb, C);
  cudaEventRecord(stop);
printf("\nCall cudaDeviceSynchronize\n");
  cudaDeviceSynchronize();

//Gpu vers cpu
  int error3 = cudaMemcpy(T_out, C, n*sizeof(int), cudaMemcpyDeviceToHost);
  if(error3)
    printf("error3 %d (Gpu vers cpu)\n",error3);

//Time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\nTime GPU : %f ms\n",milliseconds);

printf("\n T_in : \n");
  for(int i=0;i<n;i++){
printf("%d\t",T_in[i]);
  }


printf("\n T_out %d : \n",n);
  for(int i=0;i<n;i++){
printf("%d\t",T_out[i]);
  }
printf("\n");
  char s;
  printf("Press enter to continue\n");
  s=getchar();
  putchar(s);
  system("clear");
printf("\nFree - End\n");
//free
  free(T_in);
  free(T_out);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

// return 0
  return 0;
}

