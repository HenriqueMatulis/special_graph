# special_graph
Python library to compute the special case of the graph isomorphism problem where the graph can be 
sort of "embedded in ℤ x ℤ". By this I mean that each node can be thought of as being a point `(a, b)` in ℤ x ℤ
and it is neighbours with all nodes of the form `(a + da, b + db)` where `|da| + |db| = 1`.

Then to get the graph isomorphism, we simply do represent both graphs as matrices using the above correspondance and 
check the matrices against each other accounting for all symmetries of the cube.

Example usage
```
>>> import special_graph

>>> adj_list1 = {1: {(1, 0): 2},  2: {(0, 1): 3}, 3: {(-1, 0): 4}}
>>> adj_list2 = {3: {(1, 0): 1},  1: {(0, 1): 4}, 4: {(-1, 0): 2}}
>>> graph1 = special_graph.SpecialGraph(adj_list1)
>>> graph2 = special_graph.SpecialGraph(adj_list2)
>>> sparse_matrices1 = graph1.to_sparse_matrices()
>>> sparse_matrices2 = graph2.to_sparse_matrices()
>>> print(sparse_matrices1)
[{(0, 0): 1, (1, 0): 2, (1, 1): 3, (0, 1): 4}]
>>> print(sparse_matrices2)
[{(0, 0): 1, (-1, 0): 3, (0, 1): 4, (-1, 1): 2}]
>>> sparse_matrix1 = sparse_matrices1[0]
>>> sparse_matrix2 = sparse_matrices2[0]
>>> print(sparse_matrix1)
     y     
x    1    4
     2    3
>>> print(sparse_matrix2)
     y    
x    3    2
     1    4
>>> print(special_graph.matrix_equal(sparse_matrix1.to_matrix(), sparse_matrix2.to_matrix()))
{1: 3, 4: 2, 2: 1, 3: 4}
```
