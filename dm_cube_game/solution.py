from __future__ import annotations
import heapq
from typing import Union

import numpy as np
import cubeGame as cb
from collections import deque


class Node:
    """Une classe de noeuds d'un arbre ou graphe de recherche"""
    def __init__(self, plateau: cb.Plateau,
                 pred: Union[Node, None] = None,
                 action: str = "None") -> None:
        """On mémorise dans le noeud
        - l'état du jeu correspondant : le plateau
        - le noeud précédent
        - l'action qui a mené dans cet état"""
        self._plateau = plateau
        self._predecesseur = pred
        self._action = action
        # la liste des actions possibles est les 4 directions
        # ici, je mémorise les 4 fonctions qui instancient ces directions
        # on peut accéder à une représentation sous forme de string
        # d'une fonction par l'attribut __name__
        self._actions = [plateau.haut, plateau.bas,
                         plateau.droite, plateau.gauche]

    def _solution_rec(self, acc: list = []) -> list:
        """On a besoin d'un accumulateur pour implanter la récursion
        qui calcule le chemin de la racine à ce noeud"""
        if self._predecesseur == None:
            return ["None"]
        else:
            return self._solution_rec(self._predecesseur) + [self._action.__name__] 

    def solution(self) -> list:
        """Calcul du chemin de la racine au noeud"""
        node = self
        res =[] 
        while node._predecesseur:
            res = [node._action.__name__] + res
            node = node._predecesseur
        return ["None"] + res

    def __iter__(self):
        """Retourne un itérateur sur les actions de ce noeud.
        On peut donc écrire for "action in node"...
        """
        return self._actions.__iter__()

    def final(self):
        """Un noeud est final si le plateau est final"""
        return self._plateau.final()

    def __eq__(self, __o: object) -> bool:
        """Deux noeuds sont égaux si ils ont le même plateau"""
        if not isinstance(__o, Node):
            return NotImplemented
        return self._plateau == __o._plateau
    

    def __hash__(self) -> int:
        """On ne peut stocker que des valeurs hachables comme clef
        dans les dictionnaires (dict) ou ensembles (set).
        Par défaut, seuls les non mutables sont hachables.
        Mais ici un noeud ne doit pas être modifié, on va le
        hacher selon la représentation textuelle de son plateau
        (cela peut être coûteux si on avait un grand plateau...)"""
        return hash(self._plateau.__repr__())


    def __lt__(self, __o: object) -> bool:
        if not isinstance(__o, Node):
            return NotImplemented
        return True


def recherche_largeur(start: cb.Plateau) -> Union[None, list[str]]:
    """Parcours en largeur"""
    a_visiter = deque([Node(start)])
    visites = set()
    stop = False
    while not stop:
        node = a_visiter.popleft()
        visites.add(node.__hash__())
        if node._plateau.final():
            stop = True
        else:
            for action in node._actions:
                try:
                    child = Node(action(), node, action)
                    if child.__hash__() not in visites and child not in a_visiter:
                        a_visiter.append(child)
                except:
                    pass
    return node.solution()


def recherche_profondeur_bornee(start: cb.Plateau, limite_profondeur : int = 10) -> Union[None, list[str]]:
    node_start = (Node(start), 0)
    a_visiter = deque([node_start])
    visites = set()
    stop = False
    while not stop:
        node, profondeur = a_visiter.pop()
        visites.add(node.__hash__())
        if node._plateau.final():
            stop = True
        elif (profondeur < limite_profondeur):
            for action in node._actions:
                try:
                    child = Node(action(), node, action)
                    if child.__hash__() not in visites and child not in a_visiter:
                        a_visiter.append((child, profondeur+1))
                except:
                    pass
    return node.solution()


def heuristique(node):
    return 6-np.count_nonzero(node._plateau.cube.vals)

def Astar(start: cb.Plateau) -> Union[None, list[str]]:
    a_visiter = []
    heapq.heappush(a_visiter, (0, Node(start)))
    distances ={Node(start).__hash__() : 0}
    stop = False
    while not stop and len(a_visiter)>0:
        node = heapq.heappop(a_visiter)[1]
        if node._plateau.final():
            stop = True
        else:
            node_hash = node.__hash__()
            for action in node._actions:
                try:
                    child = Node(action(), node, action)
                    child_hash = child.__hash__()
                    if child_hash not in distances or distances[node_hash]+1 < distances[child_hash]:
                        distances[child_hash] = distances[node_hash]+1
                        heapq.heappush(a_visiter, (distances[child_hash] + heuristique(child), child))
                except:
                    pass
    return node.solution()


if __name__ == "__main__":

    plateau = cb.Plateau()
    plateau = plateau.droite()
    plateau = plateau.droite()
    solution_largeur = recherche_largeur(plateau)
    print(solution_largeur)
    solution_profondeur_bornee = recherche_profondeur_bornee(plateau)
    print(solution_profondeur_bornee)
    solution_astar= Astar(plateau)
    print(solution_astar)


