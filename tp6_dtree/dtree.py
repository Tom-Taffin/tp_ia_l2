from typing import Optional


class DatasetException(BaseException):
    pass


def gini(nb_total: int, nb: int) -> float:
    """computes the gini score for the binary case.
    calcule le score de gini pour un échantillon contenant nb_total
    éléments dont nb sont de classe 1 et le reste de classe 0.
    """
    if nb_total == 0:
        return 0.0
    
    p = nb/nb_total
    return 2*p*(1-p)


class Dataset:
    """ a matrix of binary values represented as 0 and 1 
      columns is the list of column names
      attributes is the set of column indexes
    """
    def load(self, filename: str) -> None:
        """loads a file, containing a comma separated list of values. 
        The first line contains the column names. All other values are binary : 0 or 1. 
        Raises an DatasetException if the format in incorrect """
        with open(filename) as f:
            self.columns = [v.strip() for v in f.readline().strip().split(',')]
            self.attributes = set(range(len(self.columns)-1))
            n = len(self.columns)
            self.data = []
            for line in f:
                vec = []
                for v in line.strip().split(","):
                    v = int(v)
                    if v==0 or v==1:
                        vec.append(v)
                if len(vec) != n:
                    raise DatasetException("Incorrect line", line, self.columns, vec)
                self.data.append(vec)

    def sort(self, start: int, end: int, attribute: int) -> int:
        """puts all samples such that attributes is 0 before all samples such that attribute is 1, 
        between start and end. Returns the index where we find the latest 0"""
        first1 = start
        for i in range(start, end):
            if not self.data[i][attribute]:
                self.data[i], self.data[first1] = self.data[first1], self.data[i]
                first1 += 1
        return first1



    def get_nb_pos(self, a: int, start: int, end: int) -> tuple[int, int, int]:
        """ returns the number of samples between start and end 
            - with target 1 and attribute a=0 and 
            - with target 1 and attribute a=1 and 
            - with attribute a = 1 """
        c1a0 = 0
        c1a1 = 0
        a1 = 0
        for i in range(start, end):
            if self.data[i][-1]:
                if self.data[i][a]:
                    c1a1 += 1
                else:
                    c1a0 += 1
            if self.data[i][a]:
                a1 += 1
        return c1a1, c1a0, a1

    def majority(self, start: int, end: int) -> int:
        """ returns the majority class of samples between start and end  """
        nb1 = 0
        for i in range(start, end):
            if self.data[i][-1]:
                nb1+=1
        if nb1>((end-start)//2):
            return 1
        return 0


    def same_class(self, start: int, end: int) -> bool:
        """returns True iff all the samples are of the same class between start and end"""
        res = True
        classe = self.data[start][-1]
        i = start+1
        while res and i<end:
            if self.data[i][-1] != classe:
                res = False
            i+=1
        return res


class Node:

    def __init__(self, dataset: Dataset,
                 start: Optional[int] = 0,
                 end: Optional[int] = None,
                 attributes: Optional[set[int]] = None,
                 level: Optional[int] = None) -> None:
        """attributes est un ensemble d'indices, ce constructeur en fait une copie.
        En effet attributes est un set, donc un mutable. L'instance pourra modifier cet ensemble. 
        Et donc l'ensemble doit être spécifique à ce noeud et non partagé avec les autres noeuds 
        (ce qui pourrait être le cas si aucune copie n'est effectuée)
        """
        self.dataset = dataset
        if start is None:
            self.start = 0
        else:
            self.start = start
        if end is None:
            self.end = len(dataset.data)
        else:
            self.end = end
        if attributes:
            self.attributes = attributes.copy()
        else:
            self.attributes = dataset.attributes.copy()
        if level is None:
            self.level = 1
        else:
            self.level = level
        
    def score_gini(self, attribute: int) -> float:
        """ computes the gini score for the dataset and attribute as 
        P[x_a=1]Gini(P[y=1|x_a=1]) + P[x_a=0]Gini(P[y=1|x_a=0])
        """
        pa, pna, a = self.dataset.get_nb_pos(attribute, self.start, self.end)
        n = self.end-self.start
        return (a/n) * gini(a, pa) + (1-a/n) * gini(n-a, pna)

    def set_best_attribute(self) -> None:
        """ computes the best attribute """
        min = None
        for att in self.attributes:
            score = self.score_gini(att)
            if min == None or score<min:
                min = score
                self.best_attribute = att

    def build_tree(self, max_level: Optional[int] = None) -> None:
        if self.dataset.same_class(self.start,self.end) or len(self.attributes) == 0:
            self.label = self.dataset.majority(self.start, self.end)
            return None
        if max_level != None and self.level >= max_level:
            self.label = self.dataset.majority(self.start, self.end)
            return None
        self.set_best_attribute()
        best = self.best_attribute

        i = self.dataset.sort(self.start, self.end, best)
        if i == self.start or i == self.end:
            self.label = self.dataset.majority(self.start, self.end)
            return None
        
        self.attributes.remove(best)        
        self.label = self.dataset.majority(self.start, self.end)
        self.tree0 = Node(self.dataset, self.start, i, self.attributes, self.level + 1)
        self.tree1 = Node(self.dataset, i, self.end, self.attributes, self.level + 1)
        
        self.tree0.build_tree(max_level)
        self.tree1.build_tree(max_level)
            

    def __str__(self) -> str:
        try:
            return "x_{}=0? (classe {}):\n{}{}\n{}{}".format(self.best_attribute,
                                                             self.label,
                                                             "  " * self.level,
                                                             self.tree0,
                                                             "  " * self.level,
                                                             self.tree1)
        except AttributeError:
            return "{}: classe {}".format(" " * self.level,
                                   self.label)


def dtree(filename: str) -> Node:
    dataset = Dataset()
    dataset.load(filename)
    n = Node(dataset)
    n.build_tree()
    return n    


    
if __name__ == "__main__":
    d = Dataset()
    d.load("test_large.data")
    print(d.data)
    print(d.attributes)
    # print(d)

    print(dtree("test_large.data"))


