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
Implémenter l'algorithme de merge and sort (https://www.cc.gatech.edu/~bader/papers/GPUMergePath-ICS2012.pdf) sur GPU + une utilisation de votre choix du merge and sort dans une application.

Introduction
------------
Naviguer dans les menus pour tester toutes nos implémentations de tri merge and sort sur CPU et GPU.
Nos applications de l'algorithme de tri :
- Test sur tableau random : génère aléatoirement un tableau d'entier et le tri
- Lecture à partir d'un fichier : lit à partir d'un fichier un tableau d'entier et le tri.
  Le fichier doit être écrit de la meme manière que data/exemple.txt
  On peut utiliser script/genere.c pour en créer un nouveau.
- Exemple d'application donnee automobile : des données automobiles sont lues et triées.
  Il est essentiel de lancer le script de pré-traitrement des données script/doc.py avant.

Bibliothèques
------------
- Il faut posséder un environement avec GPU (compilateur nvcc).
- Biblioteques C++
- Python Pandas

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

