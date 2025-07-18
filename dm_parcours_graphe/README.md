# Rendu DM Andrieu Taffin

## Parcour en largeur
Start : 1, End : 30  
![graphe parcour en largeur](img/largeur1.png)
![plot parcour en largeur](img/largeur2.png)  

## Parcour en profondeur bornée
Start : 1, End : 30  
![graphe parcour en profondeur](img/profondeur1.png)
![plot parcour en profondeur](img/profondeur2.png)  

## Parcour Astar
Start : 1, End : 30  
![graphe parcour Astar](img/astar1.png)
![plot parcour Astar](img/astar2.png)  

## Parcour RBFS
Start : 1, End : 30  
![graphe parcour Astar](img/rbfs1.png)
![plot parcour Astar](img/rbfs2.png)  

En moyenne, sur un chemin optimal long, les parcours Astar et RBFS sont plus rapide que le parcours en largeur. Sur un chemin optimal cours, le parcours en largeur devient le plus rapide. Dans les 2 cas, le parcours en profondeur est le plus lent.
Pour le cas du chemin optimal long, les parcours Astar et RBFS l'emportent car ils parcourent beaucoup moins de noeuds que le parcours en largeur.
Pour le cas du chemin optimal court, la différence du nombre de noeud parcourus n'est pas significative et le parcours en largeur devient alors plus rapide car les calculs d'heuristiques prennent du temps.
