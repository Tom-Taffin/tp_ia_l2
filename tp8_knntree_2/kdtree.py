from __future__ import annotations
from collections.abc import Callable
from collections import Counter
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class DatasetIter:
    def __init__(self, dataset: Dataset) -> None:
        self.index = 0
        self.dataset = dataset

    def __iter__(self) -> DatasetIter:
        return self

    def __next__(self) -> tuple[list[float], str]:
        ind = self.index
        self.index += 1
        if ind >= self.dataset.size:
            raise StopIteration
        return self.dataset.data[ind], self.dataset.target[ind]        

class Dataset:
    """ A simple class for data sets. It contains 
    a list of examples (x,y) where x is a data point and y is a class (target).
    We have the following attributes: 
    - size: the number of records
    - data: a list of lists of floats, point coordinates, aka x
    - target: a list of str, classes, aka y.
    - ind_iter: the index of the iteration over the data set."""
    def __init__(self) -> None:
        self.size = 0
        self.data = []
        self.target = []
        self.ind_iter = 0

    def load(self, filename: str) -> None:
        """loads a file. We assume that the file is well formatted, i.e. 
        all lines contain the same number of floats, terminated by a str in a CSV file. """
        with open(filename) as f:
            for line in f:
                liste = [v for v in line.strip().split(',')]
                self.data.append([float(v) for v in liste[:-1]])
                self.target.append(liste[-1])
                self.size += 1

    def from_data(self, dataset: Dataset, indexes: list[int]) -> None:
        """loads data from another dataset. Not optimal, It duplicate records!
        The data and target values at the indexes are inserted in this data set.
        """
        for i in indexes:
            self.data.append(dataset.data[i])
            self.target.append(dataset.target[i])
            self.size += 1

    def split(self, prop: float=0.75, seed: int=0) -> tuple[Dataset, Dataset]:
        """Splits a data set in 2 parts and returns a couple of datasets d1, d2 
        where the size of d1 is prop times the size of the current data set"""
        d1 = Dataset()
        d2 = Dataset()
        indexes = list(range(self.size))
        random.seed(seed)
        random.shuffle(indexes)
        n = math.floor(prop*self.size)
        for i in indexes[:n]:
            d1.data.append(self.data[i])
            d1.target.append(self.target[i])
            d1.size +=1
        for i in indexes[n:]:
            d2.data.append(self.data[i])
            d2.target.append(self.target[i])
            d2.size +=1
        return d1,d2

    def __iter__(self) -> Dataset:
        """The iterator definition. It will give all couples (x, y) in the data set"""
        # initialization of the index needed to enumerate all couples
        return DatasetIter(self)
    
    def sort(self, start: int, end: int, attribute: int) -> int:
        if (end-start > 1):
            pivot = self.data[start][attribute]
            dg = start+1
            for i in range(start+1, end):
                if self.data[i][attribute] < pivot:
                    self.data[dg], self.data[i] = self.data[i], self.data[dg]
                    dg += 1
            self.data[start], self.data[dg-1] = self.data[dg-1], self.data[start]
            self.sort(start, dg-1, attribute)
            self.sort(dg, end, attribute)

    def __str__(self) -> str:
        """ A string representation of the data set"""
        res = ""
        if self.size < 10:
            for ind in range(self.size):
                str_data = ", ".join([str(j) for j in self.data[ind]])
                res = res + f"{str(ind)}: {str_data}, {str(self.target[ind])}\n"
        else:
            for ind in range(5):
                str_data = ", ".join([str(j) for j in self.data[ind]])
                res = res + f"{str(ind)}: {str_data}, {str(self.target[ind])}\n"
            res += "...\n"
            for ind in range(self.size-5, self.size):
                str_data = ", ".join([str(j) for j in self.data[ind]])
                res = res + f"{str(ind)}: {str_data}, {str(self.target[ind])}\n"
        return res


def euclidean_distance(x1:list[float], x2: list[float]) -> float:
    """Compute the Euclidean distance between 2 points"""
    return math.sqrt(sum([(x2[i] - x1[i])**2 for i in range(len(x1))]))


class Node:
    def __init__(self, point, classe, sag = None, sad = None):
        self.point = point
        self.classe = classe
        self.sag = sag
        self.sad = sad
    
    def __repr__(self):
        return str(self.point) + f" ({self.classe})"

class Kdtree:
    
    def __init__(self, dataset: Dataset,
                 distance: Callable[[list[float], list[float]], float]= euclidean_distance
                 ) -> None:
        self.dataset = dataset
        self.distance = distance
        self.k = len(dataset.data[0])
        self.tree = self.build(0, dataset.size)

    def build(self, debut, fin, profondeur=0):
        if debut >= fin:
            return None

        axe = profondeur % self.k
        self.dataset.sort(debut, fin, axe)
        m = (debut + fin) // 2

        pointM = self.dataset.data[m]
        classeM = self.dataset.target[m]

        noeud = Node(
            pointM,
            classeM,
            self.build(debut, m, profondeur+1),
            self.build(m+1, fin, profondeur+1)
        )
        
        return noeud


    def search(self, point, node=None, depth=0, best=None, best_dist=float('inf')):
        """Recherche le plus proche voisin d'un point donné dans le Kdtree."""
        if node is None:
            node = self.tree

        if node is None:
            return None, float('inf')

        axis = depth % self.k

        current_dist = self.distance(point, node.point)

        if current_dist < best_dist:
            best, best_dist = node, current_dist

        if point[axis] < node.point[axis]:  
            best_node = node.sag
            other_node = node.sad
        else:
            best_node = node.sad
            other_node = node.sag

        if best_node is not None:
            best, best_dist = self.search(point, best_node, depth + 1, best, best_dist)

        if other_node is not None:
            if abs(point[axis] - node.point[axis]) < best_dist:
                best, best_dist = self.search(point, other_node, depth + 1, best, best_dist)

        return best, best_dist
    
    def search_n_nearest(self, point, n=1, node=None, depth=0, best_nodes=None, max_dist=float('inf')):
        """Recherche les n plus proches voisins d'un point donné dans le Kdtree"""
        if node is None:
            node = self.tree
            
        if best_nodes is None:
            best_nodes = []
        
        if node is None:
            return best_nodes
        
        axis = depth % self.k
        
        current_dist = self.distance(point, node.point)
        
        if len(best_nodes) < n:
            best_nodes.append((node, current_dist))
            best_nodes.sort(key=lambda x: x[1])
            max_dist = best_nodes[-1][1] if best_nodes else float('inf')
        elif current_dist < max_dist:
            best_nodes.append((node, current_dist))
            best_nodes.sort(key=lambda x: x[1])
            best_nodes = best_nodes[:n]
            max_dist = best_nodes[-1][1]
        
        if point[axis] < node.point[axis]:
            first_branch = node.sag
            second_branch = node.sad
        else:
            first_branch = node.sad
            second_branch = node.sag
        
        if first_branch is not None:
            best_nodes = self.search_n_nearest(point, n, first_branch, depth + 1, best_nodes, max_dist)
            max_dist = best_nodes[-1][1] if len(best_nodes) == n else float('inf')
        
        if second_branch is not None:
            hyperplane_dist = abs(point[axis] - node.point[axis])
            if hyperplane_dist < max_dist or len(best_nodes) < n:
                best_nodes = self.search_n_nearest(point, n, second_branch, depth + 1, best_nodes, max_dist)
        
        return best_nodes




    def plot_tree(self, ax, node, depth=0, x_min=0, x_max=10, y_min=0, y_max=10):
        """Trace les frontières du KD-tree avec des couleurs variant selon la profondeur."""
        if node is None:
            return

        colors = ['r', 'g', 'b', 'm', 'c', 'y']
        color = colors[depth % len(colors)]

        axis = depth % self.k
        x, y = node.point

        if axis == 0:
            ax.plot([x, x], [y_min, y_max], linestyle='--', color=color, linewidth = 1)
            self.plot_tree(ax, node.sag, depth + 1, x_min, x, y_min, y_max)
            self.plot_tree(ax, node.sad, depth + 1, x, x_max, y_min, y_max)
        else:
            ax.plot([x_min, x_max], [y, y], linestyle='--', color=color, linewidth = 1)
            self.plot_tree(ax, node.sag, depth + 1, x_min, x_max, y_min, y)
            self.plot_tree(ax, node.sad, depth + 1, x_min, x_max, y, y_max)

    
    
    def majority(self, neighbors: list[tuple[Node,float]]) ->  str:
        """Return the majority class in the list of neighbors"""
        occ = {}
        max = 0
        for elt in neighbors:
            if elt[0].classe in occ:
                occ[elt[0].classe] += 1
            else:
                occ[elt[0].classe] = 1
            if occ[elt[0].classe] > max:
                max = occ[elt[0].classe]
                res = elt[0].classe
        return res
    
    def predict(self, z: list[float], n) ->  str:
        """Returns the predicted class associated with z"""
        neighbours = self.search_n_nearest(z, n)
        return self.majority(neighbours)

    
    def score(self, dataset: Dataset, n: int) -> float:
        """Computes the score of this knn on a data set. This is the accuracy."""
        res = 0 
        for i in range(dataset.size):
            if self.predict(dataset.data[i],n) == dataset.target[i]:
                res += 1
        return res/dataset.size


if __name__ == '__main__':
    
    d = Dataset()
    d.load("iris.data")
    ds1, ds2 = d.split(0.10)

    

    kdtree = Kdtree(ds1)
    print(kdtree.score(ds2, 9))
    print(kdtree.score(ds1, 9))
    kdtree = Kdtree(ds2)
    print(kdtree.score(ds2, 9))
    print(kdtree.score(ds1, 9))
    

    
    
    d = Dataset()
    d.load("iris_simple2.data")
    kdtree = Kdtree(d)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = {
        "Iris-setosa" : "b",
        "Iris-versicolor" : "r",
        "Iris-virginica" : "g",
    }
    x_coords, y_coords = zip(*d.data)

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    margin = 0.5
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    kdtree.plot_tree(ax, kdtree.tree, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    
    c = [markers[y] for y in d.target]
    img = ax.scatter(x_coords, y_coords, c=c)
    
    
    point_to_search = [(x_max - x_min) * np.random.random_sample() + x_min, (y_max - y_min) * np.random.random_sample() + y_min]
    nearest_neighbors = kdtree.search_n_nearest(point_to_search, 4)
    print(kdtree.predict(point_to_search, 4))
    for neighbor in nearest_neighbors:
        ax.scatter(point_to_search[0], point_to_search[1], c="grey")
        ax.plot((point_to_search[0], neighbor[0].point[0]), (point_to_search[1], neighbor[0].point[1]))
    
    plt.show()
    