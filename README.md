# special_graph
Python library to compute the special case of the graph isomorphism problem where the graph can be 
sort of "embedded in ℤ x ℤ". By this I mean that each node can be thought of as being a point `(a, b)` in ℤ x ℤ
and its neighbours are all of the form `(a + da, b + db)` where `|da| + |db| = 1`. Then to get the graph 
isomorphism, we simply do represent both graphs as matrices using the above correspondance and 
check the matrices against each other accounting for all symmetries of the cube.
