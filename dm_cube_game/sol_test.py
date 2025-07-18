import solution
import cubeGame
import numpy as np


def prep():
    start = cubeGame.Plateau()
    start.terrain = np.full((4, 4), False)
    start.cube.vals = np.full(6, True)
    start.lig = start.col = 0

    commands = ['droite', 'droite', 'droite', 'bas', 'bas', 'bas',
                'gauche', 'gauche', 'gauche']

    terrains = [start]
    for i in range(len(commands)):
        terrains.append(terrains[i].__getattribute__(commands[i])())
    return terrains


def test_largeur():
    true_results = [["None"],
                    ["None", 'bas', 'haut', 'bas'],
                    ["None", 'gauche', 'droite'],
                    ["None", 'gauche', 'droite', 'gauche', 'gauche', 'droite'],
                    ["None", 'bas', 'haut', 'haut', 'gauche', 'gauche'],
                    ["None", 'haut', 'haut', 'gauche', 'gauche'],
                    ["None", 'haut', 'bas', 'haut', 'haut', 'haut', 'gauche',
                     'gauche'],
                    ["None", 'droite', 'haut', 'haut', 'haut', 'gauche',
                     'gauche'],
                    ["None", 'droite', 'droite', 'haut', 'haut', 'haut',
                     'gauche', 'gauche'],
                    ["None", 'haut', 'bas', 'droite', 'droite', 'droite',
                     'haut', 'haut', 'haut', 'gauche', 'gauche']]

    terrains = prep()
    results = []
    for terrain in terrains:
        results.append(solution.recherche_largeur(terrain))
    assert(results == true_results)
