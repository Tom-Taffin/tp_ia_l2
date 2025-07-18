from  knn import *


d = Dataset()
d.load("iris.data")


def test_load():
    assert d.size == 150
    assert d.target[0]=='Iris-setosa'
    for ind, elt in enumerate([5.1, 3.5, 1.4, 0.2]) :
        assert d.data[0][ind] == elt


def test_from_data():
    dout = Dataset()
    dout.from_data(d, indexes=[0, 4, 5])
    assert dout.size == 3
    assert dout.data[0] == [5.1,3.5,1.4,0.2]
    assert dout.target[0] == 'Iris-setosa'

def test_split():
    d1, d2 = d.split(prop=.5, seed=42)
    assert d1.size == d2.size == 75
    assert d2.data[0] == [6.8, 3.2, 5.9, 2.3]    
    assert d2.target[0] == "Iris-virginica"
    assert d1.data[0] == [6.1, 2.9, 4.7, 1.4]
    assert d1.target[0] == 'Iris-versicolor'

def test_euclidean_distance():
    p1 = [1.0, 4.0, 5.0, 7.0, 2.0]
    p2 = [3.0, -1.0, -1.0, -3.0, 0.0]
    assert euclidean_distance(p1, p2) == 13

def test_majority():
    classifier = Knn(d, euclidean_distance, 3)
    assert classifier.majority([(1.5, 'A'), (1, 'A'), (3, 'B')]) == 'A'

def test_neighbors():
    classifier = Knn(d, euclidean_distance, 3)
    assert classifier.predict([0, 0, 0, 0]) == "Iris-setosa"
    classifier = Knn(d, euclidean_distance, 1)
    assert classifier.predict([6.1, 2.9, 4.7, 1.4]) == "Iris-versicolor"

def test_score():
    # les données sont dans l'ordre des classes
    # prendre 80 données c'est 50 de setosa et 30 d'une autre classe
    dsimple = Dataset()
    dsimple.from_data(d, list(range(80)))
    classifier = Knn(dsimple, euclidean_distance, 1)
    assert classifier.score(dsimple) == 1
    classifier = Knn(dsimple, euclidean_distance, 75)
    assert classifier.score(dsimple) == 50/80

