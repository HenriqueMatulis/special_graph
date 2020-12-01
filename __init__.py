"""Solve a special case of the graph isomorphism problem."""
import copy
from typing import Any, List, Dict, Tuple, Optional

def _make_full_graph(graph: Dict[int, Dict[Tuple[int, int], int]]) -> Dict[int, Dict[Tuple[int, int], int]]:
    """Makes the given directed graph 'full' by adding reverse edges for every edge.

    >>> _make_full_graph({
    ...     1: {(-1, 0): 2},
    ...     2: {(-1, 0): 3}})
    {1: {(-1, 0): 2}, 2: {(-1, 0): 3, (1, 0): 1}, 3: {(1, 0): 2}}
    """
    result = copy.deepcopy(graph)
    for node, adjacent_dict in graph.items():
        for diff, val in adjacent_dict.items():
            if val not in result:
                result[val] = {}
            neg_diff = tuple(-x for x in diff)
            if neg_diff in result[val] and result[val][neg_diff] != node:
                raise Exception(f'Nodes {node} and {result[val][neg_diff]} both point to node {val} along the direction {neg_diff}')
            result[val][neg_diff] = node
    return result

class SparseMatrix(dict):
    def _bounding_box(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        locs_by_dim = list(zip(*[loc for loc in self]))
        min_x, max_x = min(locs_by_dim[0]), max(locs_by_dim[0])
        min_y, max_y = min(locs_by_dim[1]), max(locs_by_dim[1])
        return ((min_x, max_x), (min_y, max_y))

    def to_matrix(self) -> List[List[Optional[Any]]]:
        """Converts this sparse matrix into a nested list

        >>> SparseMatrix({(0, 0): 0, (0, 1): 1, (0,2): 2, (1, 2): 3, (2, 2): 4}).to_matrix()
        [[0, 1, 2], [None, None, 3], [None, None, 4]]
        >>>
        """
        ((min_x, max_x), (min_y, max_y)) = self._bounding_box()
        result = [[None] * (max_y - min_y + 1) for _ in  range(max_x - min_x + 1)]
        for loc, id in self.items():
            result[loc[0] - min_x][loc[1] - min_y] = id
        return result

    def __str__(self) -> str:
        """Pretty prints this sparse matrix to a string

        >>> print(SparseMatrix({(0, 0): 0, (0, 1): 1, (0,2): 2, (1, 2): 3, (2, 2): 4}))
             y         
        x    0    1    2
                       3
                       4
        >>>
        """
        cloned = copy.deepcopy(self)
        if (0, 0) not in cloned:
            cloned[(0, 0)] = ""
        ((min_x, max_x), (min_y, max_y)) = cloned._bounding_box()
        cloned[(min_x - 1, min_y)] = "y"
        cloned[(min_x, min_y - 1)] = "x"
        matrix = cloned.to_matrix()
        padding = 5
        result = ""
        for x, lst in enumerate(matrix):
            for y, val in enumerate(lst):
                if val is None:
                    val = ""
                to_add = str(val)[:padding]
                result += to_add
                if y + 1 < len(lst):
                    result += " " * (padding - len(to_add))
            if x + 1 < len(matrix):
                result += "\n"
        return result


class SpecialGraph:
    def __init__(self, adj_list: Dict[int, Dict[Tuple[int, int], int]]):
        self.graph = _make_full_graph(adj_list)


    def to_sparse_matrices(self) -> List[SparseMatrix]:
        """Convert this special graph into a list of sparse matrices

        >>> x = SpecialGraph({
        ...     1: {(-1, 0): 2},
        ...     2: {(-1, 0): 3},
        ...     3: {(0, -1): 4},
        ...     4: {(0, -1): 5},
        ...     5: {(1, 0): 6},
        ...     6: {(1, 0): 7}})
        >>> matrices = x.to_sparse_matrices()
        >>> len(matrices)
        1
        >>> print(matrices[0])
             y         
        x    5    4    3
             6         2
             7         1
        """
        explored = set()
        unexplored = sorted([x for x in self.graph])
        matrices = []
        while len(unexplored) > 0:
            root = unexplored[0]
            matrix = SparseMatrix()
            id_to_loc = {}
            queue = [((0, 0), root)]
            while len(queue) > 0:
                loc, id = queue.pop()
                if id in explored:
                    if loc not in matrix:
                        raise Exception(f"Id {id} must be in 2 locations at once: {loc} and {id_to_loc[id]}.")
                    if matrix[loc] != id:
                        raise Exception(f"Previously visited location {loc} has id {matrix[loc]}, but we must also place id {id} there")
                    continue
                matrix[loc] = id
                id_to_loc[id] = loc
                unexplored.remove(id)
                explored.add(id)
                for diff, neighbour in self.graph[id].items():
                    queue.append((tuple(x + y for (x, y) in zip(diff, loc)), neighbour))
            matrices.append(matrix)
        return matrices


def graph_from_matrix(matrix: List[List[Optional[int]]]) -> Dict[int, Dict[Tuple[int, int], int]]:
    """ Converts the given matrix of ids into an adjacency dict representing a graph

    >>> graph = graph_from_matrix([
    ...   [1,    None,    None],
    ...   [2,    3,       None],
    ...   [None, 4,       5   ]])
    >>> graph
    {1: {(1, 0): 2}, 2: {(0, 1): 3}, 3: {(1, 0): 4}, 4: {(0, 1): 5}}
    >>> sp_graph = SpecialGraph(graph)
    >>> matrices = sp_graph.to_sparse_matrices()
    >>> len(matrices)
    1
    >>> print(matrices[0])
         y         
    x    1         
         2    3    
              4    5
    """
    result = {}
    for x, lst in enumerate(matrix):
        for y, val in enumerate(lst):
            if val is None:
                continue
            r = {}
            if x + 1 < len(matrix) and matrix[x+1][y] is not None:
                r[(1, 0)] = matrix[x+1][y]
            if y + 1 < len(lst) and matrix[x][y+1] is not None:
                r[(0, 1)] = matrix[x][y+1]
            if r != {}:
                result[val] = r
    return result

def _matrix_exact_equal(matrix1: List[List[Optional[int]]], matrix2: List[List[Optional[int]]])-> Optional[Dict[int, int]]:
    """ Checks if 2 matrices are exactly equal (except for ids), and gives the mapping between ids if they are equal

    >>> matrix1 = [
    ...   [1,    None,    None],
    ...   [2,    3,       None],
    ...   [None, 4,       5   ]]
    >>> matrix2 = [
    ...   [5,    None,    None],
    ...   [3,    4,       None],
    ...   [None, 1,       2   ]]
    >>> matrix3 = [
    ...   [5,    None,    7   ],
    ...   [3,    4,       None],
    ...   [6,    1,       2   ]]
    >>> _matrix_exact_equal(matrix1, matrix2)
    {1: 5, 2: 3, 3: 4, 4: 1, 5: 2}
    >>> _matrix_exact_equal(matrix2, matrix1)
    {5: 1, 3: 2, 4: 3, 1: 4, 2: 5}
    >>> _matrix_exact_equal(matrix1, matrix3) is None
    True
    >>> _matrix_exact_equal(matrix2, matrix3) is None
    True
    """
    n = len(matrix1)
    if n != len(matrix2):
        return None
    if n == 0:
        return {}
    m = len(matrix1[0])
    if m != len(matrix2[0]):
        return None
    if m == 0:
        return {}

    res = {}
    for i in range(n):
        for j in range(m):
            if type(matrix1[i][j]) != type(matrix2[i][j]):
                return None
            if matrix1[i][j] is None:
                continue
            res[matrix1[i][j]] = matrix2[i][j]
    return res


def _matrix_reflect(mat: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
    """Reflects a matrix.

    >>> mat = [[1, 2], [3, 4]]
    >>> _matrix_reflect(mat)
    [[3, 4], [1, 2]]
    """
    mat = copy.deepcopy(mat)
    return list(reversed(mat))


def _matrix_transpose(mat: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
    """Transposes a matrix.

    >>> mat = [[1, 2], [3, 4]]
    >>> _matrix_transpose(mat)
    [[1, 3], [2, 4]]
    """
    if len(mat) == 0:
        return []
    n, m = len(mat), len(mat[0])
    res = [[None] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            res[i][j] = mat[j][i]
    return res


def _matrix_rotate(mat: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
    """Rotates a matrix.

    >>> mat = [[1, 2], [3, 4]]
    >>> _matrix_rotate(mat)
    [[2, 4], [1, 3]]
    """
    return _matrix_reflect(_matrix_transpose(mat))


def matrix_equal(matrix1: List[List[Optional[int]]], matrix2: List[List[Optional[int]]])-> Optional[Dict[int, int]]:
    """ Checks if 2 matrices are equal up to ids and rotations/reflections, and if so gives the mapping between ids.

    >>> matrix1 = [
    ...   [1,    None,    None],
    ...   [2,    3,       None],
    ...   [None, 4,       5   ]]
    >>> matrix2 = [
    ...   [5,    None,    None],
    ...   [3,    4,       None],
    ...   [None, 1,       2   ]]
    >>> matrix3 = [
    ...   [5,    None,    7   ],
    ...   [3,    4,       None],
    ...   [6,    1,       2   ]]
    >>> matrix_equal(matrix1, matrix2)
    {1: 5, 2: 3, 3: 4, 4: 1, 5: 2}
    >>> matrix_equal(_matrix_reflect(matrix1), matrix2)
    {1: 5, 2: 3, 3: 4, 4: 1, 5: 2}
    >>> matrix_equal(_matrix_transpose(matrix1), matrix2)
    {1: 5, 2: 3, 3: 4, 4: 1, 5: 2}
    >>> matrix_equal(_matrix_rotate(matrix1), matrix2)
    {5: 5, 4: 3, 3: 4, 2: 1, 1: 2}
    >>> matrix_equal(_matrix_reflect(matrix1), matrix3) == None
    True
    >>> matrix_equal(_matrix_transpose(matrix1), matrix3) == None
    True
    >>> matrix_equal(_matrix_rotate(matrix1), matrix3) == None
    True
    """
    mat1 = matrix1
    for _ in range(4):
        res = _matrix_exact_equal(mat1, matrix2)
        if res is not None:
            return res

        res = _matrix_exact_equal(_matrix_reflect(mat1), matrix2)
        if res is not None:
            return res
        mat1 = _matrix_rotate(mat1)
    return None


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    matrix = [
        [1,    2,    None, None],
        [None, 3,    4,    5],
        [None, None, 6   , None],
        [None, None, 7   , 8   ],
    ]
    dict_ = graph_from_matrix(matrix)
    for v in dict_:
        print(v, dict_[v])
    graph = SpecialGraph(dict_)
    result_matrices = graph.to_sparse_matrices()
    print(result_matrices[0])

