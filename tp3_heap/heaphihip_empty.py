from numpy import infty

Content = int | float

class PriorityQueueError(Exception):
    pass


class Heap:
    def __init__(self) -> None:
        """ initialiser le tas et fixer les attributs"""
        self.tab = [None]
        self.heap_size = 0

    def head(self) -> Content:
        """Retourner la valeur à la racine de l'arbre (indice 1 du tableau)"""
        return self.tab[1]

    def parent(self, i) -> int:
        """retourner l'indice du père du noeud d'indice i """
        return i//2

    def left(self, i) -> int:
        """retourner l'indice du fils gauche du noeud d'indice i """
        return 2*i

    def right(self, i) -> int:
        """retourner l'indice du fils droit du noeud d'indice i"""
        return 2*i+1

    def max_heapify(self, i) -> None:
        """appliquer l'algorithme qui assure la propriété
        max_heap sur le tableau de cette instance"""
        l = self.left(i)
        r = self.right(i)
        if l <= self.heap_size and self.tab[l] > self.tab[i]:
            largest = l
        else:
            largest = i
        if r <= self.heap_size and self.tab[r] > self.tab[largest]:
            largest = r
        if largest != i:
            self.tab[i],self.tab[largest] = self.tab[largest],self.tab[i]
            self.max_heapify(largest)

    def build_max_heap(self, tab) -> None:
        """construire une maxheap à partir d'un tableau"""
        self.tab = [None] + tab
        self.heap_size = len(tab)
        for i in range(self.heap_size//2, 0, -1):
            self.max_heapify(i)

    def heapsort(self, tab) -> None:
        """Trier le tableau tab en place en O(nlogn) avec l'API
        et mettre à jour cette instance"""
        self.build_max_heap(tab)
        for i in range(self.heap_size, 1, -1):
            self.tab[i],self.tab[1] = self.tab[1],self.tab[i]
            self.heap_size -= 1
            self.max_heapify(1)
        


class PriorityQueue:
    def __init__(self) -> None:
        """Initialiser la queue de priorité. Elle utilise un tas"""
        self.heap = Heap()

    def insert(self, key) -> None:
        """Insérer un élément dans la queue et maintenir 
        la propriété d'avoir la propriété max heap du tas"""
        self.heap.heap_size += 1
        # On fait l’hypothèse que la taille tableau est plus grande
        # que la taille du tas, sinon il faut allouer plus de place au
        # tableau
        self.heap.tab.append(-infty)
        self.increase_key(self.heap.heap_size, key)

    def maximum(self) -> Content:
        """Retourner l'élément maximal de la queue"""
        return self.heap.head()

    def extract_max(self) -> Content:
        """Enlever l'élément maximal de la queue"""
        if self.heap.heap_size < 1:
            raise PriorityQueueError("heap underflow")
        max = self.heap.tab[1]
        self.heap.tab[1] = self.heap.tab[self.heap.heap_size]
        self.heap.heap_size -= 1
        self.heap.tab.pop()
        self.heap.max_heapify(1)
        return max

    def increase_key(self, i, key) -> None:
        """Modifier la valeur d'un élément à l'indice i 
        en la remplaçant par key"""
        if key < self.heap.tab[i]:
            raise PriorityQueueError("new key is smaller than current key")
        self.heap.tab[i] = key
        while i > 1 and self.heap.tab[self.heap.parent(i)] < self.heap.tab[i]:
            self.heap.tab[i],self.heap.tab[self.heap.parent(i)] = self.heap.tab[self.heap.parent(i)],self.heap.tab[i]
            i = self.heap.parent(i)


if __name__ == "__main__":
    from random import randint

    h = Heap()
    h.build_max_heap([1, 5, 3, 6, 20, 50, 3, 2])
    print(h.tab)
    h.build_max_heap(list(range(10, 1, -1)))
    print(h.tab)

    h.heapsort([1, 5, 3, 6, 20, 50, 3, 2])
    print(h.tab)

    q = PriorityQueue()
    for _ in range(10):
        x = randint(1, 10)
        q.insert(x)
        print(x, q.heap.tab)
    print("dequeue")
    for _ in range(10):
        top = q.extract_max()
        print(top, q.heap.tab)

    try:
        q.extract_max()
    except PriorityQueueError as e:
        print(e)

    print("enqueue")
    for _ in range(10):
        x = randint(1, 10)
        q.insert(x)
        print(x, q.heap.tab)
