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
#include <stdlib.h>
#include <math.h>


#define NB 1
#define NTPB 4
#define N NTPB*NB

#define TAILLE 13
#define GRAIN 2

//Fct test tri
bool is_sorted(int* array, int n) {
  for(int i=0; i<n-1; i++) {
    if(array[i]>array[i+1]) return false;
  }
  return true;
}

//Fct test egale
bool is_equal(int* array1, int* array2, int n) {
  for(int i=0; i<n-1; i++) {
    if(array1[i]!=array2[i]) return false;
  }
  return true;
}

//Fct Merge CPU.
void MergeCPU(int *A, int *L, int leftCount, int *R, int rightCount) {
  int i = 0, j = 0, k = 0;
  while(i<leftCount && j<rightCount) {
    if(L[i] < R[j]) 
      A[k++] = L[i++];
    else A[k++] = R[j++];
  }
  while(i<leftCount) 
    A[k++] = L[i++];
  while(j<rightCount) 
    A[k++] = R[j++];
}

// Fct Merge and sort pour CPU
void mergeAndSortRecuCPU(int *A,int n) {
  int mid,i, *L, *R;
  if(n < 2) 
    return; // s'il y a moins de deux elements, on ne fait rien

  mid = n/2;

  // creation tableaux de gauche et de droite
  // de 0 à mid - 1 = gauche, il y a mid elements
  // de mid à n-1 = droite, il y a n-mid elements
  L = (int*)malloc(mid*sizeof(int)); 
  R = (int*)malloc((n-mid)*sizeof(int)); 

  for(i = 0;i<mid;i++) 
    L[i] = A[i]; 
  for(i = mid;i<n;i++) 
    R[i-mid] = A[i];

  mergeAndSortRecuCPU(L,mid);  // tri tableau de gauche
  mergeAndSortRecuCPU(R,n-mid);  // tri tableau de droite
  MergeCPU(A,L,mid,R,n-mid);  // fusion des tableaux
  free(L);
  free(R);
}



//Fct Pgcd (pour la gestion du nombre de thread)
// Source : https://openclassrooms.com/forum/sujet/algorithme-de-calcul-de-pgcd-20803
int get_pgcd(int a, int b){

  int pgcd = 0;
  while(1){
    pgcd = a % b;
      if(pgcd == 0){
        pgcd = b;
        break;
      }
    a = b;
    b = pgcd;
  }
  return pgcd;
}


//Fct Merge GPU
__host__ __device__ void mergeGPU(int* A, int na, int aid, int* B, int nb, int bid, int* C, int cid, int T) {

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

//Fct GPU partitionning GPU
__global__ void partitionningGPU(int* A, int na, int* B, int nb, int* C){
  int a, b, offset;
  int aid;
  int bid;

  int tid = blockIdx.x*blockDim.x+threadIdx.x;// identifiant de thread
  int index = tid*(na+nb)/(blockDim.x * gridDim.x);// index de debut dans C
  int a_top = (index>na)? na:index;
  int b_top = (index>na)? index-na:0;
  int a_bot = b_top;
  if(tid != 0) {
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
  }else{
    aid = 0;
    bid = 0;
  }
// merge
printf("[%d] Call merge\n",tid);
  mergeGPU(A, na, aid, B, nb, bid, C, index, (na+nb)/(blockDim.x * gridDim.x));
}

// Fct Merge and sort pour GPU
void mergeAndSortRecuGPU(int T[], int i_debut, int i_fin, int blockSize)
{
  int n = i_fin - i_debut + 1;
/*
printf("\n ------ i_debut %d i_fin %d blockSize %d \n",i_debut,i_fin, blockSize);
 printf("\n T %d : \n",n);
   for(int i=0;i<n;i++){
  printf("%d\t",T[i]);
    }
  printf("\n");
*/
  if (n < GRAIN){

    int i_milieu = i_debut + (i_fin - i_debut) / 2;

    int na = 1 + i_milieu - i_debut;
    int nb = i_fin - i_milieu;
    
    mergeAndSortRecuGPU(T, i_debut, i_milieu, blockSize);
    mergeAndSortRecuGPU(T, i_milieu+1, i_fin,blockSize);
  
    int *A, *B, *C;
    cudaMalloc(&A, na*sizeof(int));
    cudaMalloc(&B, nb*sizeof(int));
    cudaMalloc(&C, n*sizeof(int));

    if(!A || !B || !C ) {
      printf("[Error] memory alloc error\n");
      return;
    }

  //Cpu vers Gpu
    int error1 = cudaMemcpy(A, T, na*sizeof(int), cudaMemcpyHostToDevice);
    int error2 = cudaMemcpy(B, T+na, nb*sizeof(int), cudaMemcpyHostToDevice);
    if(error1)
      printf("[Error] error1 %d (Cpu vers Gpu)\n",error1);
    if(error2)
      printf("[Error] error2 %d (Cpu vers Gpu)\n",error2);

  //partitionning
printf("\nCall GPUpartitionning NB %d NTPB %d na %d nb %d\n",NB,NTPB,na,nb);
    partitionningGPU<<<NB,NTPB>>>(A, na, B, nb, C);

  //Gpu vers cpu
    int error3 = cudaMemcpy(T, C, n*sizeof(int), cudaMemcpyDeviceToHost);
    if(error3)
      printf("[Error] error3 %d (Gpu vers cpu)\n",error3);

  //free
    free(T);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

  }else{
  //tri
    mergeAndSortRecuCPU(T, n);
  }
}


// Fct intro
void intro(){
//INTRODUCTION Longue
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
  printf("\n");
  sleep(1);
  printf("Version Beta 2\n");
  sleep(1);

  system("clear");
  char s;
  printf("Press enter to continue\n");
  s=getchar();
  putchar(s);
  system("clear");
}


// Fct intro
void introShort(){
//INTRODUCTION courte
  system("clear");
  printf("Dans le cadre\n");
  printf("\n");
  printf("N9-IPA PARALLELISME AVANCE HPCA\n");
  printf("Projet de fin de module\n");
  printf("\n");
  printf("2017 - 2018\n");
  sleep(2);

  system("clear");
  printf("ALOUI Driss\n");
  printf("DO Alexandre\n");
  printf("\n");
  sleep(2);

  system("clear");
  printf("Présente\n");
  sleep(1);

  system("clear");
  printf("MergeAndSortAppInCuda\n");
  printf("\n");
  printf("Version Beta 2\n");
  sleep(1);

  system("clear");
  char s;
  printf("Press enter to continue\n");
  s=getchar();
  putchar(s);
  system("clear");
}

//2 Test sur tableau random
void tabRandom(){
  int x;
  int Entree = 0;
  int Erreur = 0;
  while(Entree == 0){
    system("clear");
    std::cout<<"2\tTest sur tableau random"<<std::endl;

    if(Erreur == 1){
      std::cout<<"[Erreur] Entrée non valide ! (entier positif)"<<std::endl;
      Erreur  =  0;
    }
    std::cout<<"Entrez la taille du tableau puis valider avec la touche ENTREE"<<std::endl;
    std::cin>>x;
    if(x>0){
      Entree = 1;
      char s;
      printf("Vous avez entrez %d\n", x);
      printf("Press enter to continue\n");
      s=getchar();
      putchar(s);
    }else{
      Erreur  =  1;
    }
  }

  int n = x;

// var pour timer
  cudaEvent_t startCPU, stopCPU;
  cudaEventCreate ( &startCPU );
  cudaEventCreate ( &stopCPU );
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate ( &startGPU );
  cudaEventCreate ( &stopGPU );

//Alloc Array
printf("Alloc Array\n");
  srand(time(NULL));
  int* T_cpu = (int*)malloc(n*sizeof(int));
  int* T_gpu = (int*)malloc(n*sizeof(int));
  int cpt=0;

//init Array
printf("\nInit Array\n");
  while (cpt<n){
    T_gpu[cpt]=(rand()%100);
    T_cpu[cpt] = T_gpu[cpt];
    cpt++;
  }
printf("\n");

//sort CPU
printf("\nCall sort CPU\n");
  cudaEventRecord(startCPU);
  mergeAndSortRecuGPU(T_gpu, 0, n-1, N);
  cudaEventRecord(stopCPU);
printf("\n");
//sort GPU
printf("\nCall sort GPU\n");
  cudaEventRecord(startGPU);
  mergeAndSortRecuCPU(T_cpu, n);
  cudaEventRecord(stopGPU);
printf("Call cudaDeviceSynchronize\n");
  cudaDeviceSynchronize();

//test tri ok?
printf("\nTest tri\n");
  if(is_sorted(T_cpu, n)){
    printf("OK\n");
  //Fct test egale
printf("Test egale\n");
    if(is_equal(T_cpu, T_gpu, n))
      printf("OK\n");
    else
      printf("[error] T_gpu mal trie");
  }else{
    printf("[error] T_cpu mal trie");   
  }

//Time resuts
printf("\nTime resuts\n");
  float millisecondsGPU = 0;
  cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
  printf("\nTime GPU : %f ms\n",millisecondsGPU);
  float millisecondsCPU = 0;
  cudaEventElapsedTime(&millisecondsCPU, startCPU, stopCPU);
  printf("Time CPU : %f ms\n",millisecondsCPU);

printf("\n");
  char s;
  printf("Press enter to continue\n");
  s=getchar();
  putchar(s);

//free
  free(T_cpu);
  free(T_gpu);

}

//3 Lecture à partir d'un fichier

//4 Exemple d'application donnee automobile


int sousMenu(){
  int x;
  int Entree = 0;
  int Erreur = 0;
  while(Entree == 0){
    system("clear");
    std::cout<<"Tester l'agorithme de tri"<<std::endl;
    std::cout<<"0\tQuitter"<<std::endl;
    std::cout<<"1\tHelp"<<std::endl;
    std::cout<<"2\tTest sur tableau random"<<std::endl;
    std::cout<<"3\tLecture à partir d'un fichier"<<std::endl;
    std::cout<<"4\tExemple d'application donnee automobile"<<std::endl;
    std::cout<<"5\tRetour menu principal"<<std::endl;

    if(Erreur == 1){
      std::cout<<"[Erreur] Entrée non valide !"<<std::endl;
      Erreur  =  0;
    }
  std::cout<<"Appuyer sur une touche entre 0 et 2  puis valider avec la touche ENTREE"<<std::endl;
  std::cin>>x;

  switch ( x ){
    case 0: 
      system("clear");
      std::cout<<"Vous avez choisi de quitter"<<std::endl;
      std::cout<<"Etes vous sur?"<<std::endl;
      std::cout<<"Appuyer sur 1 pour revenir au Menu principal sinon une touche pour quitter puis valider avec la touche ENTREE"<<std::endl;
      std::cin>>x;
      if(x!=1){
        system("clear");
        std::cout<<"Merci et à bientot"<<std::endl;
        sleep(0.6);
        system("clear");
        std::cout<<"Merci et à bientot"<<std::endl;
        std::cout<<"3"<<std::endl;
        sleep(1);
        system("clear");
        std::cout<<"Merci et à bientot"<<std::endl;
        std::cout<<"  2"<<std::endl;
        sleep(1);
        system("clear");
        std::cout<<"Merci et à bientot"<<std::endl;
        std::cout<<"    1"<<std::endl;
        sleep(1);
        Entree = 2;
        system("clear");
      }
      break; 
    case 1:  
      system("clear");
      std::cout<<"1\tHelp"<<std::endl;
      std::cout<<"Merci d'utiliser exclusivement les CHIFFRES pour les selections dans les menus"<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<"Le but du programme est de....."<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<"Utilisation :"<<std::endl;
      std::cout<<"- ...."<<std::endl;
      std::cout<<"- ...."<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<""<<std::endl;
      sleep(2);
      char sh;
      printf("Press enter to continue\n");
      sh=getchar();
      putchar(sh);
      break;  
    case 2:  
      // Test sur tableau random
      tabRandom();
      break;
    case 3:  
    // Lecture à partir d'un fichier

      break;
    case 4:  
    // Exemple d'application donnee automobile

      break;
    case 5:  
      Entree = 1;
      break;
    default:  
      Erreur = 1;
    } 
  }
  return Entree;
}

void menuPrincipal(){
  int x;
  int Entree = 0;
  int Erreur = 0;
  while(Entree == 0){
    system("clear");
    std::cout<<"Menu principal"<<std::endl;
    std::cout<<"0\tQuitter"<<std::endl;
    std::cout<<"1\tHelp"<<std::endl;
    std::cout<<"2\tTester l'agorithme de tri"<<std::endl;
    std::cout<<"3\tReset"<<std::endl;

    if(Erreur == 1){
      std::cout<<"[Erreur] Entrée non valide !"<<std::endl;
      Erreur  =  0;
    }
  std::cout<<"Appuyer sur une touche entre 0 et 2  puis valider avec la touche ENTREE"<<std::endl;
  std::cin>>x;

  switch ( x ){
    case 0: 
      system("clear");
      std::cout<<"Vous avez choisi de quitter"<<std::endl;
      std::cout<<"Etes vous sur?"<<std::endl;
      std::cout<<"Appuyer sur 1 pour revenir au Menu principal sinon une touche pour quitter puis valider avec la touche ENTREE"<<std::endl;
      std::cin>>x;
      if(x!=1){
        system("clear");
        std::cout<<"Merci et à bientot"<<std::endl;
        sleep(0.6);
        system("clear");
        std::cout<<"Merci et à bientot"<<std::endl;
        std::cout<<"3"<<std::endl;
        sleep(1);
        system("clear");
        std::cout<<"Merci et à bientot"<<std::endl;
        std::cout<<"  2"<<std::endl;
        sleep(1);
        system("clear");
        std::cout<<"Merci et à bientot"<<std::endl;
        std::cout<<"    1"<<std::endl;
        sleep(1);
        Entree = 1;
        system("clear");
      }
      break; 
    case 1:  
      system("clear");
      std::cout<<"1\tHelp"<<std::endl;
      std::cout<<"Merci d'utiliser exclusivement les CHIFFRES pour les selections dans les menus"<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<"Le but du programme est de....."<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<"Utilisation :"<<std::endl;
      std::cout<<"- ...."<<std::endl;
      std::cout<<"- ...."<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<""<<std::endl;
      std::cout<<""<<std::endl;
      sleep(2);
      char sh;
      printf("Press enter to continue\n");
      sh=getchar();
      putchar(sh);
      break;  
    case 2:  
      if(sousMenu()==2)
        Entree = 1;
      break;
    case 3:  
      intro();
      break;
    default:  
      Erreur = 1;
    } 
  }
}


//main
int main(){
  //int n = TAILLE;

// Gestion de l intro
  FILE* fichier = NULL;
  fichier = fopen("tmp.txt", "r");

  if (fichier == NULL){ // 1ere fois
    intro();
    FILE* fichier2 = NULL;
    fichier2 = fopen("tmp.txt", "w");
    fprintf(fichier2, "1");
    fclose(fichier2);
  }else{ // intro deja faite une fois)
    fclose(fichier);
    introShort();
  }

menuPrincipal();

/*
// var pour timer
  cudaEvent_t startCPU, stopCPU;
  cudaEventCreate ( &startCPU );
  cudaEventCreate ( &stopCPU );
  cudaEvent_t startGPU, stopGPU;
  cudaEventCreate ( &startGPU );
  cudaEventCreate ( &stopGPU );

//Alloc Array
printf("Alloc Array\n");
  srand(time(NULL));
  int* T_cpu = (int*)malloc(n*sizeof(int));
  int* T_gpu = (int*)malloc(n*sizeof(int));
  int cpt=0;

//init Array
printf("\nInit Array\n");
  while (cpt<n){
    T_gpu[cpt]=(rand()%100);
    T_cpu[cpt] = T_gpu[cpt];
    cpt++;
  }
printf("\n");

//sort CPU
printf("\nCall sort CPU\n");
  cudaEventRecord(startCPU);
  mergeAndSortRecuGPU(T_gpu, 0, n-1, N);
  cudaEventRecord(stopCPU);
printf("\n");
//sort GPU
printf("\nCall sort GPU\n");
  cudaEventRecord(startGPU);
  mergeAndSortRecuCPU(T_cpu, n);
  cudaEventRecord(stopGPU);
printf("Call cudaDeviceSynchronize\n");
  cudaDeviceSynchronize();

//test tri ok?
printf("\nTest tri\n");
  if(is_sorted(T_cpu, n)){
    printf("OK\n");
  //Fct test egale
printf("Test egale\n");
    if(is_equal(T_cpu, T_gpu, n))
      printf("OK\n");
    else
      printf("[error] T_gpu mal trie");
  }else{
    printf("[error] T_cpu mal trie");   
  }

//Time resuts
printf("\nTime resuts\n");
  float millisecondsGPU = 0;
  cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);
  printf("\nTime GPU : %f ms\n",millisecondsGPU);
  float millisecondsCPU = 0;
  cudaEventElapsedTime(&millisecondsCPU, startCPU, stopCPU);
  printf("Time CPU : %f ms\n",millisecondsCPU);
*/
printf("\n");
  char s;
  printf("Press enter to continue\n");
  s=getchar();
  putchar(s);
  system("clear");
printf("\nFree - End\n");
  printf("\nBy\n");
  printf("ALOUI Driss\n");
  printf("DO Alexandre\n");
  printf("\nNovembre 2017\n");
//free
  //free(T_cpu);
  //free(T_gpu);

// return 0
  return 0;
}

