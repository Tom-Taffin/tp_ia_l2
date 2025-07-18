"""

A naive implementation of KNN's. 

"""
from __future__ import annotations
from collections.abc import Callable
import csv
import math
import random
import numpy as np

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
    """ A simple class for data sets. I contains 
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
        """loads data from another dataset. Not optimal, I duplicate records!
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
    return math.sqrt(sum((x1[i]-x2[i])**2 for i in range(len(x1))))


class Knn:
    """A simple class for knn classifiers"""
    def __init__(self, dataset: Dataset,
                 distance: Callable[[list[float], list[float]], float]= euclidean_distance,
                 k: int=3) -> None:
        self.dataset = dataset
        self.distance = distance
        self.k = k

    def majority(self, neighbors: list[tuple[float, str]]) ->  str:
        """Return the majority class in the list of neighbors"""
        occ = {}
        max = 0
        for elt in neighbors:
            if elt[-1] in occ:
                occ[elt[-1]] += 1
            else:
                occ[elt[-1]] = 1
            if occ[elt[-1]] > max:
                max = occ[elt[-1]]
                res = elt[-1]
        return res

    def predict(self, z: list[float]) ->  str:
        """Returns the predicted class associated with z"""
        neighbors = []
        for i in range(self.dataset.size):
            neighbors.append((self.distance(z, self.dataset.data[i]), self.dataset.target[i]))
        neighbors.sort()
        return self.majority(neighbors[:self.k])
        

    def score(self, dataset: Dataset) -> float:
        """Computes the score of this knn on a data set. This is the accuracy."""
        res = 0 
        for i in range(dataset.size):
            if self.predict(dataset.data[i]) == dataset.target[i]:
                res += 1
        return res/dataset.size

if __name__ == "__main__":
    import timeit
    d = Dataset()
    d.load("iris/iris.data")
    print(d)
    ds1, ds2 = d.split(0.10)
    it = iter(ds1)
    for i in range(5):
        print(next(it))
    knn = Knn(ds1,k=9)
    print(knn.score(ds2))
    print(knn.score(ds1))
    knn = Knn(ds2,k=9)
    print(knn.score(ds2))
    print(knn.score(ds1))
    print(timeit.timeit('knn.score(d)', globals=globals(), number=10))
