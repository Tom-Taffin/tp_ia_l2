"""

An implementation of KNN's. 

"""
from __future__ import annotations
from typing import Optional
from collections.abc import Callable
import csv
import math
import random
import numpy as np


def euclidean_distance(x1:list[float], x2: list[float]) -> float:
    """Compute the Euclidean distance between 2 points"""
    res = 0.0
    for ind, val in enumerate(x1):
        res += (val - x2[ind])**2
    return math.sqrt(res)

def insert_neighbor(index: int,
                    dist: float,
                    k: int,
                    neighbors: list[tuple[float, int]]) -> None:
    """ inserts an element in the list of neighbors, ordered by increasing distance"""    
    for ind_n in range(k):
        if neighbors[ind_n][0] > dist:
            neighbors.insert(ind_n, (dist, index))
            del neighbors[-1]
            break

class DatasetIter:
    """Iterators on datasets"""
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

class KnnError(Exception):
    """Exceptions in Knn..."""

class Dataset:
    """ A simple class for data sets. I contains 
    a list of examples (x,y) where x is a data point and y is a class (target).
    We have the following attributes: 
    - size: the number of records
    - data: a list of lists of floats, point coordinates, aka x
    - target: a list of str, classes, aka y.
    - dim: the dimension of the data set."""
    def __init__(self) -> None:
        self.data = []
        self.target = []
        self.size = 0
        self.dim = 0

    def empty(self):
        """Reset the the data set to the empty set"""
        self.data = []
        self.target = []
        self.size = 0
        self.dim = 0

    def load(self, filename: str) -> None:
        """loads a file. We assume that the file is well formatted, i.e. 
        all lines contain the same number of floats, terminated by a str in a CSV file. """
        self.empty()
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0:
                    self.data.append([float(j) for j in row[:-1]])
                    self.target.append(row[-1])
        self.size = len(self.data)
        if self.size >0:
            self.dim = len(self.data[0])
        else:
            self.dim = 0

    def from_data(self, dataset: Dataset, indices: list[int]) -> None:
        """loads data from another dataset. Not optimal, I duplicate records!
        The data and target values at the indices are inserted in this data set.
        """
        self.empty()
        for ind in indices:
            self.data.append(dataset.data[ind])
            self.target.append(dataset.target[ind])
        self.size = len(indices)
        if self.size >0:
            self.dim = len(self.data[0])
        else:
            self.dim = 0

    def split(self, prop: float=0.75, seed: int=0) -> tuple[Dataset, Dataset]:
        """Splits a data set in 2 parts and returns a couple of datasets d1, d2 
        where the size of d1 is prop times the size of the current data set"""
        random.seed(seed)
        split_index = int(self.size*prop)
        interval = list(range(self.size))
        random.shuffle(interval)
        d1 = Dataset()
        d1.from_data(self, interval[:split_index])
        d2 = Dataset()
        d2.from_data(self, interval[split_index:])

        return d1, d2

    def get_neighbors(self,
                      z: list[float],
                      distance: Callable[[list[float], list[float]], float],
                      k:int) -> list[int]:
        """Returns the list of k neighbors of z in the data set """
        neighbors = [(np.inf, -1) for _ in range(k)]
        for ind in range(self.size):
            x = self.data[ind]
            dist_z = distance(x, z)
            if dist_z > neighbors[-1][0]:
                continue
            # we will insert a new neighbor !
            insert_neighbor(ind, dist_z, k, neighbors)
        return [ind for _,ind in neighbors]

    def __iter__(self) -> DatasetIter:
        """The iterator definition. It will give all couples (x, y) in the data set"""
        # initialization of the index needed to enumerate all couples
        return DatasetIter(self)

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
    
    def sort(self, start: int, end: int, attribute: int) -> int:
        if (end-start > 1):
            pivot = self.data[start][attribute]
            dg = start+1
            for i in range(start+1, end):
                if self.data[i][attribute] < pivot:
                    self.data[dg], self.data[i] = self.data[i], self.data[dg]
                    self.target[dg], self.target[i] = self.target[i], self.target[dg]
                    dg += 1
            self.data[start], self.data[dg-1] = self.data[dg-1], self.data[start]
            self.target[start], self.target[dg-1] = self.target[dg-1], self.target[start]
            self.sort(start, dg-1, attribute)
            self.sort(dg, end, attribute)

class KdTree:
    """Non optimal implementation of Kd-trees... 
    
    This class represents a node of the Kd-tree
    - dataset: the data set associated with the Kdtree
    - indices: list of indices in the dataset associated with the node
               (note that leaves of the trees have an empty list of indices or only 1 indice)
    - parent: parent node
    - axis: the associated axis (coordinate)
    - depth: depth of the node

    we also memorize the index of the data point associated with the node in an attribute 'node', and the left and right nodes when it is not a leaf...

    See https://fr.wikipedia.org/wiki/Recherche_des_plus_proches_voisins#Partitionnement_de_l'espace
     """
    def __init__(self, 
                 dataset: Dataset, 
                 indices: list[int],
                 parent:Optional[KdTree] = None,
                 axis:int=0, 
                 depth: int=0) -> None:
        self.dataset = dataset
        self.indices = indices
        self.axis = axis
        self.depth = depth
        self.parent = parent
        self._build()

    def _build(self) -> None:
        """build the kd-tree. Note that when there are only 2 indices, one is at the root, the other of the left and the right node is empty """
        #  0. check the 2 cases to stop the recursion depending on the size of the indices
        # otherwise, 
        #  1. we sort indices according to the axis coordinate of the corresponding data
        #  2. we memorize the index of the point corresponding to the median
        #  3. we build the two nodes at the next depth (named self.left and self.right)
        if len(self.indices)==0:
            self.node = None
            self.right = None
            self.left = None
        elif len(self.indices)==1:
            self.node = self.indices[0]
            self.right = None
            self.left = None
        else:
            self.axis = self.depth % self.dataset.dim
            self.indices.sort(key=lambda ind: self.dataset.data[ind][self.axis])
            m = len(self.indices)//2
            self.node = self.indices[m]
            self.left = KdTree(self.dataset,self.indices[:m],self,self.axis, self.depth+1)
            self.right = KdTree(self.dataset,self.indices[m+1:],self,self.axis, self.depth+1)



    def down(self,
              z: list[float],
              distance: Callable[[list[float], list[float]], float],
              k:int
              ) -> KdTree:
        """It goes down to find the neighbors. Returns the node at the leaf of the corresponding branch """
        # 0. if it is empty, it is an error! 
        # 1. if it is of length 1, it is the node
        # 2. if it is of legnth 2, it is the left child
        # 3. otherwise we need to select the right branch and go down
        if len(self.indices) == 0:
            raise Exception("it is empty")
        if len(self.indices) == 1:
            return self
        if len(self.indices) == 2:
            return self.left
        if self.dataset.data[self.node][self.axis]<z[self.axis]:
            return self.right.down(z,distance,k)
        return self.left.down(z,distance,k)
        


    def _up(self,
            z: list[float],
            current: KdTree,
            distance: Callable[[list[float], list[float]], float],
            k:int,
            neighbors: list[tuple[float, int]],
            explored: set[KdTree]
            ) -> None:
        """It goes up (and down) to find the neighbors. It updates the list of neighbors. It updates the set of explored nodes """
        # The most difficult part...
        # 0. the current node is explored
        # 1. do we need to add the current node in the neighbors?
        # 2. do we cut the hyperplan ?
        # 3. if so we go down in the other branch
        # 4. we go up if the parent node has not been explored
        
        
        current_dist = distance(current.dataset.data[current.node], z)
        
        explored.add(current)

        if current_dist < neighbors[-1][0]:
            i=0
            while (neighbors[i][1]!=-1 and neighbors[i][0]<current_dist):
                i+=1
            if (neighbors[i][1]==-1):
                neighbors[i] = (current_dist,current.node)
                neighbors.sort(key=lambda x: x[0])
            else :
                tmp = neighbors[i]
                neighbors[i] = (current_dist,current.node)
                i+=1
                while(i<len(neighbors)):
                    tmp2 = neighbors[i]
                    neighbors[i] = tmp
                    tmp = tmp2
                    i+=1

        hyperplane_dist = abs(current.dataset.data[current.node][current.axis] - z[current.axis])
        if current.left != None and current.left.node != None and current.left not in explored and hyperplane_dist < neighbors[-1][0]:
            node_down = current.left.down(z, distance, k)
            self._up(z,node_down,distance, k, neighbors, explored)
        elif current.right != None and current.right.node != None and current.right not in explored and hyperplane_dist < neighbors[-1][0]:
            node_down = current.right.down(z, distance, k)
            self._up(z,node_down,distance, k, neighbors, explored)
        if current.parent != None and current.parent not in explored:
            self._up(z,current.parent,distance,k,neighbors,explored)

    def get_neighbors(self,
                      z: list[float],
                      distance: Callable[[list[float], list[float]], float],
                      k:int) -> list[int]:
        """Returns the list of k neighbors of z in the data set """
        # go down
        node_down = self.down(z, distance, k)
        ind_down = node_down.node
        dist = distance(z, self.dataset.data[ind_down])
        neighbors = [(dist, ind_down)] + [(np.inf, -1) for _ in range(k-1)]
        # go up
        if node_down.parent is not None:
            explored = {node_down}
            # print(f"down {node_down.node}")
            self._up(z, node_down.parent, distance, k, neighbors, explored)
        # return the indices
        return [ind for _, ind in neighbors]

    def __str__(self) -> str:
        indent = "  " * (self.depth + 1)
        if self.node is None:
            return "empty"
        try:
            left_str = str(self.left) if self.left else "empty"
            right_str = str(self.right) if self.right else "empty"
            return f"node {self.dataset.data[self.node]}(Ind {self.node}; Axis {self.axis})\n{indent}{left_str}\n{indent}{right_str}"
        except AttributeError:
            return f"node {self.dataset.data[self.node]}(Ind {self.node}; Axis {self.axis})"


class Knn:
    """A simple class for knn classifiers"""
    def __init__(self, dataset: Dataset,
                 distance: Callable[[list[float], list[float]], float]= euclidean_distance,
                 k: int=3,
                 use_kdtree: bool=False) -> None:
        self.k = k
        self.dataset = dataset
        self.distance = distance
        self.use_kdtree = use_kdtree
        if use_kdtree:
            self.kdtree = KdTree(self.dataset, list(range(self.dataset.size)))

    def majority(self, neighbors: list[int]) ->  str:
        """Return the majority class in the list of neighbors"""
        stats = {}
        max_c = ""
        max_val = 0
        for ind in neighbors:
            c = self.dataset.target[ind]
            stats[c] = stats.get(c, 0) + 1
            if stats[c] > max_val:
                max_c = c
                max_val = stats[c]

        return max_c

    def predict(self, z: list[float]) ->  str:
        """Returns the predicted class associated with z"""
        if self.use_kdtree:
            neighbors = self.kdtree.get_neighbors(z, self.distance, self.k)
        else: 
            neighbors = self.dataset.get_neighbors(z, self.distance, self.k)
        return self.majority(neighbors)

    def score(self, dataset: Dataset) -> float:
        """Computes the score of this knn on a data set. This is the accuracy."""
        errors = 0
        for z, y in dataset:
            if y != self.predict(z):
                errors += 1
        return 1 - errors / dataset.size

if __name__ == "__main__":
    import timeit
    d = Dataset()
    d.load("iris/simple.csv")
    kd = KdTree(d, list(range(d.size)), None)
    print(kd)
    print(kd.get_neighbors([0,0], euclidean_distance, 3))

    d = Dataset()
    d.load("iris/iris.data")
    print(d)
    ds1, ds2 = d.split(0.10)
    knn = Knn(ds1,k=9,use_kdtree=True)
    print(knn.score(ds2))
    print(knn.score(ds1))
    knn = Knn(ds2,k=9, use_kdtree=True)
    print(knn.score(ds2))
    print(knn.score(ds1))
    print(timeit.timeit('knn.score(d)', globals=globals(), number=10))
