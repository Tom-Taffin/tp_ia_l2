"""

A naive implementation of KNN's with kd-trees. 

"""
from __future__ import annotations
from collections.abc import Callable
import csv
import math
import random
import numpy as np



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


class Dataset:
    """ A simple class for data sets. I contains 
    a list of examples (x,y) where x is a data point and y is a class (target).
    We have the following attributes: 
    - size: the number of records
    - data: a list of lists of floats, point coordinates, aka x
    - target: a list of str, classes, aka y.
    - dim: the dimension of the data set."""
    def __init__(self) -> None:
        self.empty()


    def empty(self):
        """Reset the the data set to the emptyset"""
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
        self.dim = len(self.data[0])

    def from_data(self, dataset: Dataset, indexes: list[int]) -> None:
        """loads data from another dataset. Not optimal, I duplicate records!
        The data and target values at the indexes are inserted in this data set.
        """
        self.empty()
        for ind in indexes:
            self.data.append(dataset.data[ind])
            self.target.append(dataset.target[ind])
        self.size = len(indexes)
        self.dim = len(self.data[0])

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


def euclidean_distance(x1:list[float], x2: list[float]) -> float:
    """Compute the Euclidean distance between 2 points"""
    res = 0.0
    for ind, val in enumerate(x1):
        res += (val - x2[ind])**2
    return math.sqrt(res)


class Knn:
    """A simple class for knn classifiers"""
    def __init__(self, dataset: Dataset,
                 distance: Callable[[list[float], list[float]], float]= euclidean_distance,
                 k: int=3) -> None:
        self.k = k
        self.dataset = dataset
        self.distance = distance

    def majority(self, neighbors: list[tuple[float,  str]]) ->  str:
        """Return the majority class in the list of neighbors"""
        stats = {}
        max_c = ""
        max_val = 0
        for _, c in neighbors:
            stats[c] = stats.get(c, 0) + 1
            if stats[c] > max_val:
                max_c = c
                max_val = stats[c]

        return max_c

    def predict(self, z: list[float]) ->  str:
        """Returns the predicted class associated with z"""
        neighbors = [(np.inf, "") for _ in range(self.k)]
        for x, y in self.dataset:
            distance = self.distance(x, z)
            if distance > neighbors[-1][0]:
                continue
            # we will insert a new neighbor !
            for ind in range(self.k):
                if neighbors[ind][0] > distance:
                    neighbors.insert(ind, (distance, y))
                    del neighbors[-1]
                    break
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
