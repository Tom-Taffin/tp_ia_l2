"""

A naive implementation of KNN's. 

"""
from __future__ import annotations
from collections.abc import Callable
from collections import Counter
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

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
        pass

    def load(self, filename: str) -> None:
        """loads a file. We assume that the file is well formatted, i.e. 
        all lines contain the same number of floats, terminated by a str in a CSV file. """
        with open(filename) as f:
            self.data = []
            self.target = []
            for line in f:
                vec = []
                for v in line.strip().split(","):
                    if is_float(v):
                        vec.append(float(v))
                    else:
                        self.target.append(v)
                self.data.append(vec)
            self.size = len(self.data)

    def from_data(self, dataset: Dataset, indexes: list[int]) -> None:
        """loads data from another dataset. Not optimal, It duplicate records!
        The data and target values at the indexes are inserted in this data set.
        """
        self.data = []
        self.target = []
        for i in indexes:
            self.target.append(dataset.target[i])
            self.data.append(dataset.data[i])
        self.size = len(self.data)

    def split(self, prop: float=0.75, seed: int=0) -> tuple[Dataset, Dataset]:
        """Splits a data set in 2 parts and returns a couple of datasets d1, d2 
        where the size of d1 is prop times the size of the current data set"""
        d1, d2 = Dataset(), Dataset()

        d1.size = int(prop * self.size)
        d2.size = int(self.size - d1.size)

        random.seed(seed)
        indices = [i for i in range(self.size)]
        random.shuffle(indices)

        d1.data = []
        d1.target = []
        for i in range(d1.size):
            d1.data.append(self.data[indices[i]])
            d1.target.append(self.target[indices[i]])

        d2.data = []
        d2.target = []
        for i in range(d2.size):
            d2.data.append(self.data[indices[d1.size+i]])
            d2.target.append(self.target[indices[d1.size+i]])
        
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
    return math.sqrt(sum([(x2[i] - x1[i])**2 for i in range(len(x1))]))


class Knn:
    """A simple class for knn classifiers"""
    def __init__(self, dataset: Dataset,
                 distance: Callable[[list[float], list[float]], float]= euclidean_distance,
                 k: int=3) -> None:
        self.dataset = dataset
        self.distance = distance
        self.k = k

    def majority(self, neighbors: list[tuple[float, list[float],  str]]) ->  str:
        """Return the majority class in the list of neighbors"""
        return Counter([n[-1] for n in neighbors]).most_common(1)[0][0]

    def predict(self, z: list[float]) ->  str:
        """Returns the predicted class associated with z"""
        distances = []
        for i in range(len(self.dataset.data)):
            data = self.dataset.data[i]
            dist = self.distance(z, data)
            distances.append((dist, data, self.dataset.target[i]))
        distances.sort(key=lambda x: x[0])
        return self.majority(distances[:self.k])

    def score(self, dataset: Dataset) -> float:
        """Computes the score of this knn on a data set. This is the accuracy."""
        nb_correct = 0
        it = iter(dataset)
        for _ in range(dataset.size):
            x, y_true = next(it)
            y_prediction = self.predict(x)
            if y_prediction == y_true:
                nb_correct += 1
        return nb_correct/dataset.size
    
    def predict_plot(self, z: list[float]) ->  str:
        distances = []
        for i in range(len(self.dataset.data)):
            data = self.dataset.data[i]
            dist = self.distance(z, data)
            distances.append((dist, data, self.dataset.target[i]))
        distances.sort(key=lambda x: x[0])
        for d in distances[:self.k]:
            ax.plot([z[0], d[1][0]], [z[1], d[1][1]], [z[2], d[1][2]])
        return self.majority(distances[:self.k])


if __name__ == "__main__":
    import timeit
    d = Dataset()
    d.load("iris.data")
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    markers = {
        "Iris-setosa" : "b",
        "Iris-versicolor" : "r",
        "Iris-virginica" : "g",
    }
    x, y, z, s = zip(*ds2.data)
    s = np.array(s)*20
    c = [markers[y] for y in ds2.target]

    img = ax.scatter(x, y, z, s=s, c=c)

    knn = Knn(ds2,k=9)
    p = ds1.data[3]
    pred = knn.predict_plot(p)
    ax.scatter(p[0], p[1], p[2], s=p[3]*20, c=markers[pred], edgecolors="k")
    
    plt.show()

