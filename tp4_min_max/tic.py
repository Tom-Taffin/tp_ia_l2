import numpy as np


class Board:
    """ Jeu sans IA """
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.render = self.board.reshape((3,3))

    def actions(self):
        return np.flatnonzero(self.board == 0)

    def trans(self, act, player):
        self.board[act] = player
        
    def undo(self, act):
        self.board[act] = 0
    
    def final(self):
        return len(self.actions()) == 0 or self.winner() 

    def winner(self):
        return ( np.any(np.abs(self.render.sum(axis=0)) == 3) or
                 np.any(np.abs(self.render.sum(axis=1)) == 3) or
                 np.abs(self.board[::4].sum()) == 3 or
                 np.abs(self.board[2::2].sum()) == 3)
    
    def utility(self, player):
        if self.winner():
            return -player
        else:
            return 0

    def display(self):
        print(self.render)

if __name__ == "__main__":
    b = Board()
    player = 1
    while True:
        b.display()
        actions = b.actions()
        print(f"Player {player}; actions : {actions}")
        pos = int(input("pos ? "))

        if pos not in actions:
            print("impossible move")
            break
        b.trans(pos, player)
        if b.final():
            if b.winner():
                print(f"Player {player} wins!")
            else:
                print("Tie!")
            break
        player = - player
    

