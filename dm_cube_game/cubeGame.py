from __future__ import annotations

import numpy as np

# Face sur le terrain
FACE = 1
# parties du dé qui changent de position quand il roule
# à droite ou à gauche
IGD = np.arange(4)  # les indices 0, 1, 2, 3
IHB = np.array([1, 4, 3, 5])


class Cube:
    """Un cube dont les indices sont
      4
    0 1 2 3
      5
    """
    def __init__(self) -> None:
        """Le cube a 6 faces stockées dans l'attribut vals.
        Au début, toutes les faces sont à True"""
        self.vals = np.full(6, True)

    def __eq__(self, __o: object) -> bool:
        """Teste si deux cubes sont identiques"""
        # Les tests d'égalité ne sont implantés
        # qu'entre deux instances de Cube
        if not isinstance(__o, Cube):
            return NotImplemented
        # Deux cubes sont égaux si leurs faces sont
        # identiques et posés de la même façon
        return np.array_equal(self.vals, __o.vals)

    def __repr__(self) -> str:
        return "\n\t{}\n{}\t{}\t{}\t{}\n\t{}".format(self.vals[4],
                                                     self.vals[0],
                                                     self.vals[1],
                                                     self.vals[2],
                                                     self.vals[3],
                                                     self.vals[5])

    def haut(self) -> None:
        """le cube roule vers le haut"""
        self.vals[IHB] = np.roll(self.vals[IHB], -1)

    def bas(self) -> None:
        """Le cube roule vers le bas"""
        self.vals[IHB] = np.roll(self.vals[IHB], 1)

    def gauche(self) -> None:
        """Le cube roule à gauche"""
        self.vals[IGD] = np.roll(self.vals[IGD], 1)

    def droite(self) -> None:
        """le cube roule à droite"""
        self.vals[IGD] = np.roll(self.vals[IGD], -1)
    
    def copy(self) -> object:
        """Retourne une copie du cube"""
        new_cube = Cube()
        new_cube.vals = self.vals.copy()
        return new_cube


class HorsPlateau(Exception):
    """Une exception à lancer si on tente de rouler
    hors du plateau"""
    # Rien à faire, c'est juste pour permettre de faire
    # des try et capture l'exception de bon type
    pass


class Plateau:
    """Un plateau de jeu est un terrain et un cube. On mémorise la
    position du cube"""

    def __init__(self, do_alea=False, seed=None, long_alea=20) -> None:
        """crée un plateau avec un terrain carré de côté 4 et un cube.
        Le cube est par défaut en ligne 0 et colonne 0.
        Si do_alea est True alors le cube est positionné au hasard
        sur le terrain et roule pendant long_alea mouvements aléatoires.
        """
        self.terrain = np.full((4, 4), False)
        self.cube = Cube()
        self.lig = self.col = 0
        if do_alea:
            rng = np.random.RandomState(seed)
            self._alea(rng, long_alea)

    def copy(self) -> Plateau:
        """Pour copier un plateau on duplique le cube et le terrain
        et on positionne le cube au même endroit sur le terrain."""
        new_plateau = Plateau()
        new_plateau.terrain = self.terrain.copy()
        new_plateau.cube = self.cube.copy()
        new_plateau.lig = self.lig
        new_plateau.col = self.col
        return new_plateau

    def __eq__(self, __o: object) -> bool:
        """Test d'égalité entre deux plateaux. Ils sont égaux
        s'ils ont les mêmes valeurs sur le terrain et le cube et
        si le cube est positionné au même endroit sur les terrains."""
        if not isinstance(__o, Plateau):
            return NotImplemented
        # Deux cubes sont égaux si leurs faces sont
        # identiques et posés de la même façon
        return np.array_equal(self.terrain, __o.terrain) and self.cube == __o.cube and self.lig == __o.lig and self.col == __o.col

    def __repr__(self) -> str:
        """représentation du terrain pour l'affichage par print"""
        return self.terrain.__str__() +\
            f"\nDé en ({self.lig},{self.col})" +\
            self.cube.__repr__()

    def _alea(self, rng, long_alea) -> None:
        """le cube est positionné au hasard sur le terrain et
        roule pendant long_alea mouvements aléatoires."""
        self.lig = rng.randint(4)
        self.col = rng.randint(4)
        i = 0
        p = self
        while i < long_alea:
            try:
                actions = [p.bas, p.haut, p.gauche, p.droite]
                p = actions[rng.randint(0, 4)]()
                i += 1
            except HorsPlateau:
                pass
        self.lig = p.lig
        self.col = p.col
        self.terrain = p.terrain
        self.cube = p.cube

    def _update_face(self) -> None:
        """les valeurs de la face du dé et de la case de terrain correspondante
           sont échangées
        """
        self.cube.vals[1] , self.terrain[self.lig][self.col] = self.terrain[self.lig][self.col] , self.cube.vals[1]

    def haut(self) -> Plateau:
        """ Le cube roule vers le haut.
        Le plateau (self) n'est pas modifié, mais un nouveau
        plateau est retourné avec le terrain et le cube, sa position
        modifiés par le déplacement.

        Une exception HorsPlateau est lancée si le cube devait
        sortir du plateau et le cube reste là où il est."""
        if self.lig == 0:
            raise HorsPlateau()
        else:    
            plateau = self.copy()
            plateau.cube.haut()
            plateau.lig -=1
            plateau._update_face()
            return plateau

    def bas(self) -> Plateau:
        """ Le cube roule vers le bas 
        Le plateau (self) n'est pas modifié, mais un nouveau
        plateau est retourné avec le terrain et le cube, sa position
        modifiés par le déplacement.

        Une exception HorsPlateau est lancée si le cube devait
        sortir du plateau et le cube reste là où il est."""
        if self.lig == 3:
            raise HorsPlateau()  
        else:       
            plateau = self.copy()
            plateau.cube.bas()
            plateau.lig +=1
            plateau._update_face()
            return plateau

    def droite(self) -> Plateau:
        """ Le cube roule vers la droite 
        Le plateau (self) n'est pas modifié, mais un nouveau
        plateau est retourné avec le terrain et le cube, sa position
        modifiés par le déplacement.

        Une exception HorsPlateau est lancée si le cube devait
        sortir du plateau et le cube reste là où il est.
        """
        if self.col == 3:
            raise HorsPlateau()  
        else:       
            plateau = self.copy()
            plateau.cube.droite()
            plateau.col +=1
            plateau._update_face()
            return plateau

    def gauche(self) -> Plateau:
        """ Le cube roule vers la gauche 
        Le plateau (self) n'est pas modifié, mais un nouveau
        plateau est retourné avec le terrain et le cube, sa position
        modifiés par le déplacement.

        Une exception HorsPlateau est lancée si le cube devait
        sortir du plateau et le cube reste là où il est.
        """
        if self.col == 0:
            raise HorsPlateau() 
        else:        
            plateau = self.copy()
            plateau.cube.gauche()
            plateau.col -=1
            plateau._update_face()
            return plateau

    def final(self) -> bool:
        """C'est gagné quand le terrain est vide"""
        return not np.any(self.terrain)


if __name__ == "__main__":
    t = Plateau(seed=4)
    print(t)
    t = Plateau(seed=np.random.randint(100))
    print(t)
    d = Cube()
    d.vals = np.array([False, True, False, False, False, False])
    d.gauche()
    print(d)
