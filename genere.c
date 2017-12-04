#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(){
	int nombre_aleatoire = 0;
	FILE* fichier3 = NULL;
	fichier3 = fopen("exemple2.txt", "w");
	int i;
	for(i=0; i<10000; i++) {
		nombre_aleatoire = rand();
		fprintf(fichier3,"%d\n",nombre_aleatoire );
	}
	fclose(fichier3);
	return 0;
}
