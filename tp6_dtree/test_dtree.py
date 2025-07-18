import dtree
import pytest

def test_load():
    d = dtree.Dataset()
    d.load("test.data")
    assert(d.columns == ['A', 'B', 'C', 'target'])
    assert(d.attributes == {0, 1, 2})
    assert(d.data==[[0, 1, 0, 0], [1, 0, 1, 1]])

def test_gini():
    assert(dtree.gini(10,0)==0)
    assert(dtree.gini(10,5) == 0.5)
    assert(dtree.gini(10,4) > dtree.gini(10,2))


def test_load_bad():
    with pytest.raises(dtree.DatasetException):
        d = dtree.Dataset()
        d.load("test_bad.data")


def test_sort():
    d = dtree.Dataset()
    d.load("test_large.data")
    n = len(d.data)
    r = d.sort(0, n, 0)
    assert(r == 2)
    assert(d.data[0] == [0, 1, 1, 0, 0])
    assert(d.data[1] == [0, 1, 1, 0, 1])
    r = d.sort(0, n, 2)
    assert(r == 0)
    r = d.sort(0, n, 3)
    assert(r == n)
    d.load("test_large.data")
    r = d.sort(4, n, 0)
    assert(r == 5)
    assert(d.data[0] == [0, 1, 1, 0, 0])
    assert(d.data[4] == [0, 1, 1, 0, 1])


def test_get_nb_pos():
    d = dtree.Dataset()
    d.load("test_large.data")
    pa, pna, a = d.get_nb_pos(1, 0, 5)
    assert(pa==0)
    assert(pna==2)
    assert(a==3)
    pa, pna, a = d.get_nb_pos(1, 5, 10)
    assert(pa==2)
    assert(pna==2)
    assert(a==3)

def test_majority():
    d = dtree.Dataset()
    d.load("test_large.data")
    assert(d.majority(0,5)==0)
    assert(d.majority(0,4)==0)
    assert(d.majority(0,7)==1)


def test_same_class():
    d = dtree.Dataset()
    d.load("test_large.data")
    assert(d.same_class(0,2)==False)
    assert(d.same_class(0,1)==True)
    assert(d.same_class(7,7)==True)
    assert(d.same_class(5,8)==True)
    assert(d.same_class(5,9)==False)


def test_score_gini():
    d = dtree.Dataset()
    d.load("test_large.data")
    n = dtree.Node(d)
    assert(n.score_gini(0)==0.475)
    assert(n.score_gini(2)==0.48)
    assert(n.score_gini(3)==0.48)


def test_set_best_attribute():
    d = dtree.Dataset()
    d.load("test_large.data")
    n = dtree.Node(d)
    n.set_best_attribute()
    assert(n.best_attribute==1)