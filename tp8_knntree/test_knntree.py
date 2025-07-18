import knntree as knn

d = knn.Dataset()
d.load("iris/iris.data")

dsimple = knn.Dataset()
dsimple.load("iris/simple.csv")

def test_load():
    "load dataset"
    assert d.size == 150
    assert d.target[0]=='Iris-setosa'
    for ind, elt in enumerate([5.1, 3.5, 1.4, 0.2]) :
        assert d.data[0][ind] == elt


def test_from_data():
    "new dataset from 0, 4, 5"
    dout = knn.Dataset()
    dout.from_data(d, indices=[0, 4, 5])
    assert dout.size == 3
    assert dout.data[0] == [5.1,3.5,1.4,0.2]
    assert dout.target[0] == 'Iris-setosa'

def test_split():
    "we split: 2 equal parts"
    d1, d2 = d.split(prop=.5, seed=42)
    assert d1.size == d2.size == 75
    assert d2.data[0] == [6.8, 3.2, 5.9, 2.3]    
    assert d2.target[0] == "Iris-virginica"
    assert d1.data[0] == [6.1, 2.9, 4.7, 1.4]
    assert d1.target[0] == 'Iris-versicolor'

def test_euclidean_distance():
    "simple distance test"
    p1 = [1.0, 4.0, 5.0, 7.0, 2.0]
    p2 = [3.0, -1.0, -1.0, -3.0, 0.0]
    assert knn.euclidean_distance(p1, p2) == 13

def test_majority():
    "virginica are at indexes 100..."
    classifier = knn.Knn(d, knn.euclidean_distance, 3)
    assert classifier.majority([1, 102, 101]) == 'Iris-virginica'

def test_neighbors():
    " neighbors simple"
    classifier = knn.Knn(d, knn.euclidean_distance, 3)
    assert classifier.predict([0, 0, 0, 0]) == "Iris-setosa"
    classifier = knn.Knn(d, knn.euclidean_distance, 1)
    assert classifier.predict([6.1, 2.9, 4.7, 1.4]) == "Iris-versicolor"

def test_score():
    "score of 1 nn and full-nn"
    # les données sont dans l'ordre des classes
    # prendre 80 données c'est 50 de setosa et 30 d'une autre classe
    d80 = knn.Dataset()
    d80.from_data(d, list(range(80)))
    classifier = knn.Knn(d80, knn.euclidean_distance, 1)
    assert classifier.score(d80) == 1
    classifier = knn.Knn(d80, knn.euclidean_distance, 75)
    assert classifier.score(d80) == 50/80

def test_buildkdtree():
    "test on build"
    kd = knn.KdTree(dsimple, list(range(dsimple.size)))
    assert kd.node == 1

def test_down():
    "down"
    kd = knn.KdTree(dsimple, list(range(dsimple.size)))
    assert kd.down([-1.0, 4.0], knn.euclidean_distance, 1).node == 3
    assert kd.down([1.0, 4.0], knn.euclidean_distance, 1).node == 4

def test_kd_get_neighbors():
    "get neighbors for kd-trees"
    kd = knn.KdTree(dsimple, list(range(dsimple.size)))
    assert kd.get_neighbors([-1, -1], knn.euclidean_distance, 1) == [2]
    assert kd.get_neighbors([0, 0], knn.euclidean_distance, 1) == [1]
    assert kd.get_neighbors([0,0], knn.euclidean_distance, 3) == [1, 2, 4]

    kd = knn.KdTree(d, list(range(d.size)))
    assert kd.get_neighbors(d.data[0], knn.euclidean_distance, 5) == [0, 17, 4, 39, 28]