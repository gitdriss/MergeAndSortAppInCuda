MergeAndSortAppInCuda
===
    N9-IPA PARALLELISME AVANCE HPCA
    ABBAS TURKI Lokman

Date: 19/10/2017

    ALOUI Driss
    DO Alexandre
    
MAIN 5 Polytech Paris UPMC

Consignes
------------
Implémenter l'algo de merge and sort (https://www.cc.gatech.edu/~bader/papers/GPUMergePath-ICS2012.pdf) sur GPU + une utilisation de votre choix du merge and sort dans une application.

Introduction
------------
Naviguez dans les menus pour tester toutes nos implementations de tri merge and sort sur CPU et GPU.
Nos applications de l'algorithme de tri :
- Test sur tableau random : genere aleatoirement un tableau d'entier et le tri
- Lecture à partir d'un fichier : lit à patir d'un fichier un tableau d'entier et le tri.
  Le fichier doit etre ecrit de la meme maniere que data/exemple.txt
  On peut utiliser script/genere.c pour en creer un nouveau.
- Exemple d'application donnee automobile : des donnees automobiles sont lues et triees.
  Il est essentiel de lancer le script de pretraitrement des donnees script/doc.py avant.

Bibiotheques
------------
- Il faut posseder un environement avec GPU (compilateur nvcc).
- Biblioteques C++
- Python Panda

Utilisation
------------

    Make
    Make exec
    Make clean
    Make reset
    
Make reset suprime les fichiers data.

Exemple de tests
------------

N: 5463 GRAIN: 10

    Time GPU : 3.856832 ms
    Time CPU : 6.504000 ms
    

N: 100 GRAIN: 10

    Time GPU : 0.141312 ms
    Time CPU : 0.109536 ms

Sources
------------
https://www.cc.gatech.edu/~bader/papers/GPUMergePath-ICS2012.pdf

Remerciement
------------

    Adrian Ahne
    David Danieli
    Louis Aldebert
    Lucas Gaudelet
    Paul-Alexis Dray

