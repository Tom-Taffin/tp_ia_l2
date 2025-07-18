from board import *
import heapq
from collections import deque
import itertools
counter = itertools.count()     # unique sequence count

class Node:
    def __init__(self, board: Board, parent=None, action=None):
        self.board = board
        self.state = board.tab.tobytes()
        self.parent = parent
        self.action = action
        if self.parent is None:
            self.cost = 0
        else:
            self.cost = self.parent.cost + 1
        self.heuristic = self.board.heuristic()
    
    def act(self, action):
        return Node(self.board.move(action), self, action)

    def _get_solution(self, sol: list) -> list:
        if self.parent is None:
            return sol
        else:
            sol.insert(0, self.action)
            return self.parent._get_solution(sol)

    def get_actions(self) -> list:
        return self.board.admissible_moves()

    def get_solution(self) -> list:
        sol = []
        return self._get_solution(sol)

    def is_goal(self) -> bool:
        return self.board.is_final()

    def __eq__(self, __o: object) -> bool:
        return self.state == __o.state

    def __hash__(self) -> int:
        return hash(self.state)

def bfs(start: Node):
    frontiere = deque([start])
    visites = set()
    stop = False
    while not stop:
        node = frontiere.popleft()
        visites.add(node.__hash__())
        if node.is_goal():
            stop = True
        else:
            for action in node.get_actions():
                child = node.act(action)
                if child.__hash__() not in visites and child not in frontiere:
                    frontiere.append(child)
    return node.get_solution()

def astar(start: Node) ->list:
    a_visiter = []
    heapq.heappush(a_visiter, (0, next(counter) , start))
    distances ={start.__hash__() : 0}
    stop = False
    while not stop:
        node = heapq.heappop(a_visiter)[2]
        if node.is_goal():
            stop = True
        else:
            node_hash = node.__hash__()
            for action in node.get_actions():
                child = node.act(action)
                child_hash = child.__hash__()
                if child_hash not in distances or distances[node_hash]+1 < distances[child_hash]:
                    distances[child_hash] = distances[node_hash]+1
                    heapq.heappush(a_visiter, (distances[child_hash] + child.heuristic, next(counter), child))
    return node.get_solution()
            
def astar_bounded(start: Node, level) ->list:
    a_visiter = []
    heapq.heappush(a_visiter, (0, next(counter) , start))
    distances ={start.__hash__() : 0}
    stop = False
    while not stop:
        node = heapq.heappop(a_visiter)[2]
        if node.is_goal():
            stop = True
        else:
            node_hash = node.__hash__()
            if distances[node_hash]<level:
                for action in node.get_actions():
                    child = node.act(action)
                    child_hash = child.__hash__()
                    if child_hash not in distances or distances[node_hash]+1 < distances[child_hash]:
                        distances[child_hash] = distances[node_hash]+1
                        heapq.heappush(a_visiter, (distances[child_hash] + child.heuristic, next(counter), child))
    return node.get_solution()

if __name__ == "__main__":
    from random import seed

    seed(0)
    b = Board()
    print(b)
    b = b.shuffle(28)
    print(b)
    start = Node(b)
    print(bfs(start))
    print(astar(start))
    print(astar_bounded(start, 13))