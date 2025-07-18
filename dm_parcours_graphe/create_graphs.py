import networkx as nx
import matplotlib.pyplot as plt
from typing import Union
from collections import deque
import numpy as np
import heapq
import time

# Fichiers de graphes sont ici : https://people.math.sc.edu/Burkardt/datasets/cities/cities.html

#____________________________________________________________________________
"""
Créé le graphe et récupère les coordonnées
"""

def create_graph(filename, limit=None):
    G = nx.Graph()
    source = 0 
    nb_nodes = None
    with open(filename) as f:
        for line in f:
            if not line.startswith("#"):
                dists = [int(i) for i in line.strip().split()]
                if nb_nodes == None:
                    nb_nodes = len(dists)
                if len(dists) != nb_nodes:
                    print("erreur", line)
                    continue
                # Je suppose que c'est bon
                for target, dist in enumerate(dists):
                    if source != target and (limit == None or limit > dist):
                        G.add_edge(source, target, weight=dist)
                source += 1
    return G

def coordinates(filename, limit=None):
    res = []
    with open(filename) as f:
        for line in f:
            if not line.startswith("#"):
                co = [float(i) for i in line.strip().split()]
                res.append(co)
    return res

#____________________________________________________________________________
"""
Les différents algorithmes de recherche
"""

def recherche_largeur(G: nx.Graph, start: int, end: int) -> Union[None, list[int]]:
    """Parcours en profondeur bornee"""
    node_start = start
    a_visiter = deque([node_start])
    visites = set()
    stop = False
    predecesseur = {start : None}
    while not stop:
        node = a_visiter.popleft()
        visites.add(node)
        if node == end:
            stop = True
        else:
            for adj in G.neighbors(node):
                if adj not in visites and adj not in a_visiter:
                    a_visiter.append(adj)
                    predecesseur[adj] = node
    return predecesseur

def recherche_profondeur_bornee(G: nx.Graph, start: int, end: int, limite_profondeur : int = 100) -> Union[None, list[int]]:
    """Parcours en profondeur bornee"""
    node_start = (start, 0)
    a_visiter = deque([node_start])
    visites = set()
    stop = False
    predecesseur = {start : None}
    while not stop:
        node, profondeur = a_visiter.pop()
        visites.add(node)
        if node == end:
            stop = True
        elif (profondeur < limite_profondeur):
            for adj in G.neighbors(node):
                if adj not in visites and adj not in a_visiter:
                    a_visiter.append((adj, profondeur+1))
                    predecesseur[adj] = node
    return predecesseur

def heuristique(s, v):
    return np.linalg.norm(np.array(coordonees[s]) - np.array(coordonees[v]))

def Astar(G: nx.Graph, start: int, end: int) -> Union[None, list[str]]:
    a_visiter = []
    heapq.heappush(a_visiter, (0, start))
    distances ={start : 0}
    stop = False
    predecesseur = {start : None}
    while not stop:
        node = heapq.heappop(a_visiter)[1]
        if node == end:
            stop = True
        else:
            for adj in G.neighbors(node):
                if adj not in distances or distances[node]+G[node][adj]['weight'] < distances[adj]:
                    distances[adj] = distances[node]+G[node][adj]['weight']
                    heapq.heappush(a_visiter, (distances[adj] + heuristique(adj,end), adj))
                    predecesseur[adj] = node
    return predecesseur

class Node:
    def __init__(self, val, pred=None, path_cost=0):
        self.val = val
        self.pred = pred
        self.path_cost = path_cost
    
    def __eq__(self, other):
        return self.val == other.val
    
    def __repr__(self):
        return str(self.val) + " " + str(self.pred.val) + " " + str(self.path_cost)

def RBFS(G : nx.Graph, node : Node, goal : Node, f_limit):

    visites = set()

    def Recursive_Best_First_Search(G : nx.Graph, node : Node, goal : Node, f_limit):
        visites.add(node.val)
        
        if node == goal:
            return node, 0
        
        successors = []
        for n in list(G.neighbors(node.val)):
            if n not in visites:
                successors.append(Node(n, node, node.path_cost+G[node.val][n]["weight"]))
        
        if len(successors) == 0:
            return  None, np.inf
        
        for s in successors:
            s.f = max(s.path_cost + heuristique(s.val, goal.val), node.f)
        
        while True:
            successors.sort(key=lambda x: x.f)
            best = successors[0]

            if best.f > f_limit:
                return None, best.f
            
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = np.inf
            
            result, best.f = Recursive_Best_First_Search(G, best, goal, min(f_limit, alternative))

            if result is not None:
                return result, best.f
    
    return Recursive_Best_First_Search(G, node, goal, f_limit)

#____________________________________________________________________________
"""
Les fonctions de chemin et de solution
"""

def chemin(predecesseur, start, end):
    c = [end]
    while end != start:
        end = predecesseur[end]
        c.append(end)
    return c

def chemin_node(node : Node):
    if not isinstance(node, Node):
        return []
    else:
        return [node.val] + chemin_node(node.pred)

def solution_largeur(G, start, end):
    return chemin(recherche_largeur(G, start, end), start, end)

def solution_profondeur_bornee(G, start, end, p=100):
    return chemin(recherche_profondeur_bornee(G, start, end, p), start, end)

def solution_Astar(G, start, end):
    return chemin(Astar(G, start, end), start, end)

def solution_rbfs(G, start, end):
    node_start = Node(start)
    node_end = Node(end)
    node_start.f = heuristique(node_start.val, node_end.val)
    result, bestf = RBFS(G, node_start, node_end, np.inf)
    return chemin_node(result)

#____________________________________________________________________________
"""
Utilisation des algos et affichage
"""

if __name__ == '__main__':
    G = create_graph("sgb128_dist.txt", limit=300)
    coordonees = coordinates("sgb128_xy.txt")
    x_coordonees, y_coordonees = [co[0] for co in coordonees], [co[1] for co in coordonees]

    start = 4
    end = 21

    print("Parcours en largeur")
    t1 = time.process_time()
    for _ in range(100):
        solution_largeur(G, start, end)
    t2 = time.process_time()
    sol_largeur = solution_largeur(G, start, end)
    print(sol_largeur)
    print("Temps :", t2-t1, "\n")
    
    print("Parcours en profondeur borné")
    t1 = time.process_time()
    for _ in range(100):
        solution_profondeur_bornee(G, start, end)
    t2 = time.process_time()
    sol_profondeur_bornee = solution_profondeur_bornee(G, start, end)
    print(sol_profondeur_bornee)
    print("Temps :", t2-t1, "\n")
    
    print("Astar")
    t1 = time.process_time()
    for _ in range(100):
        solution_Astar(G,start,end)
    t2 = time.process_time()
    sol_Astar = solution_Astar(G,start,end)
    print(sol_Astar)
    print("Temps :", t2-t1, "\n")

    print("RBFS")
    t1 = time.process_time()
    for _ in range(100):
        solution_rbfs(G, start, end)
    t2 = time.process_time()
    sol_rbfs = solution_rbfs(G, start, end)
    print(sol_rbfs)
    print("Temps :", t2-t1, "\n")

    

    sol_affiche = sol_rbfs # Ici changer la solution à afficher

    """
    couleurs = {s: "blue" for s in G.nodes()}
    for node in sol_affiche:
        couleurs[node] = "pink"
    nx.draw(G, node_color=couleurs.values())
    """
    

    plt.plot(x_coordonees, y_coordonees, 'o')
    for i in range(len(sol_affiche)-1):
        plt.plot((x_coordonees[sol_affiche[i]], x_coordonees[sol_affiche[i+1]]), 
                 (y_coordonees[sol_affiche[i]], y_coordonees[sol_affiche[i+1]]))

    plt.show()
    
    nx.write_gml(G, "sgb128_dist-300.gml")