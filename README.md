# MH
MPRO - Projet de métaheuristiques

Ronan LE PRAT - 2025

## Exécution

Pour exécuter le programme:
- Posséder une version de python supérieure ou égale à 3.13.0
- Installer les dépendances nécessaires: `pip install -r requirements.txt`
- Run le programme principal: `python ./src/main.py <file_path> <optional:heuristics> <optional:metaheuristics>`

Par exemple : `python ./src/main.py ./instances/cap71.txt [PPU] [RC]`

Le programme affiche dans la console les différentes méthodes appelées, et à chaque fois, les améliorations de la solution courante obtenues.

A la fin de l'exécution du programme, il est affiché dans le terminal la meilleure solution obtenue pour l'instance, ainsi que le temps total d'exécution.

## Paramètres d'exécution autorisés

### file_path

Chemin d'accès local vers une instance du problème de localisation discrète.

Le format est celui accessible à l'adresse suivante : https://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html.

Les capacités doivent être spécifiées (même si elles ne sont pas utilisées) par des valeurs entières.

### heuristics

Liste d'heuristiques parmi :
- PPU - Plus Proche Usine - Chaque client est affecté à l'usine la plus proche.
- CG - Construction Gloutonne - On ouvre une seule usine, puis on regarde successivement si ouvrir une autre fait baisser le coût.
- CMI - Couverture Minimum Itérée - Heuristique basée sur une réduction (partielle) du problème à un problème de couverture.

La liste des heuristiques doit être donnée au format suivant : `[HE1,...,HEn]`. Par exemple: `[CMI,PPU]`.

Par défaut, les heuristiques suivantes sont appliquées : `[PPU,CG,CMI]`.

On effectue ensuite une descente à partir de chacune des solutions données par les heuristiques d'entrées.

## metaheuristics

Liste de descentes ou métaheuristiques parmi :
- C1-H - Exploration complète du voisinage 1-Hamming.
- CP-H - Exploration successive des voisinages 1/2/3-Hamming.
- AP-H - Exploration aléatoire des voisinages 1/2/.../m-Hamming.
- RC - Recuit simulé.

La liste des métaheuristiques doit être donnée au format suivant : `[ME1,...,MEn]`. Par exemple: `[C1-H,RC]`.

Par défaut, les métaheuristiques suivantes sont appliquées : `[CP-H]`.