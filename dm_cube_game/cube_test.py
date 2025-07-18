# import pytest
import cubeGame
import numpy as np

vde1 = np.array([False, True, False, False, False, False])
vdeG = np.array([False, False, True, False, False, False])
vdeH = np.array([False, False, False, False, False, True])
vdeB = np.array([False, False, False, False, True, False])
vdeD = np.array([True, False, False, False, False, False])


def test_de_haut():
    d = cubeGame.Cube()
    d.vals = vde1.copy()
    d.haut()
    assert(np.all(d.vals == vdeH))


def test_de_gauche():
    d = cubeGame.Cube()
    d.vals = vde1.copy()
    d.gauche()
    assert(np.all(d.vals == vdeG))


def test_de_droite():
    d = cubeGame.Cube()
    d.vals = vde1.copy()
    d.droite()
    assert(np.all(d.vals == vdeD))


def test_de_bas():
    d = cubeGame.Cube()
    d.vals = vde1.copy()
    d.bas()
    assert(np.all(d.vals == vdeB))


start = np.array([[False, True, False, False],
                  [True, True, False, False],
                  [False, False, True, False],
                  [False, True, False, True]])
stepH = np.array([[False, False, False, False],
                  [True, True, False, False],
                  [False, False, True, False],
                  [False, True, False, True]])


def test_haut():
    t = cubeGame.Plateau()
    t.cube.vals[4] = True
    # on va vers le haut
    t.terrain = start.copy()
    t.col = 1
    t.lig = 1
    t = t.haut()
    assert(np.all(t.terrain == start))


def test_eq():
    p1 = cubeGame.Plateau(seed=4)
    p2 = cubeGame.Plateau(seed=4)
    assert(p1 == p2)
    p2 = p2.bas()
    assert(p1 != p2)
    p2 = p2.haut()
    assert(p1 != p2)
    p2 = p2.bas()
    p2 = p2.haut()
    assert(p1 == p2)


