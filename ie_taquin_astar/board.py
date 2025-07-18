import numpy as np
from random import randint

class BoardException(Exception):
    pass

class Board:
    def __init__(self, n=3):
        self.tab = np.arange(n ** 2).reshape((n, n))
        self.solution = self.tab.copy()
        self.hole = (0, 0)
        self.n = n

    def move(self, cell:tuple):
        l_h, c_h = self.hole
        l, c = cell
        tab = self.tab.copy()
        if c == c_h:
            if l < l_h:
                tab[l+1:l_h+1, c] = self.tab[l:l_h, c]
            elif l > l_h:
                tab[l_h:l, c] = self.tab[l_h+1:l+1, c]
            else:
                raise BoardException("Impossible move")
        elif l == l_h:
            if c < c_h:
                tab[l, c+1:c_h+1] = self.tab[l, c:c_h]
            elif c > c_h:
                tab[l, c_h:c] = self.tab[l, c_h+1:c+1]
            else:
                raise BoardException("Impossible move")
        else:
            raise BoardException("Impossible move")
        next = Board()
        next.tab = tab
        next.solution = self.solution
        next.n = self.n
        next.tab[l, c] = 0
        next.hole = cell
        return next
        

    def shuffle(self, nbr=10) -> None:
        b = self
        i = 0
        while i < nbr:
            try:
                l, c = (randint(0, self.n-1), randint(0, self.n-1))
                b = b.move((l,c))
                i += 1
            except BoardException:
                pass
        return b

    def admissible_moves(self) -> list:
        r = [(self.hole[0], c) for c in range(self.n) if c != self.hole[1]]
        r.extend([(l, self.hole[1]) for l in range(self.n) if l != self.hole[0]])
        return r

    def is_final(self) -> bool:
        return np.all(self.solution == self.tab) 

    def heuristic(self) -> int:
        res = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.tab[i,j]!=self.solution[i,j]:
                    res += 1
        return res

    def __str__(self) -> str:
        return self.tab.__str__()

    def __eq__(self, __o: object) -> bool:
        return np.all(self.tab == __o.tab)



