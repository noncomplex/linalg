import math
import numpy as np

# vector representations in python using numpy. row/col/shape field/transpose
vec_aslist = [1, 2, 3, 4, 5]  #np.shape(vec_aslist): (5,)
vec_asarray = np.array([1, 2, 3, 4, 5]) # .shape property returns tuples: (5,)
vec_asnprow = np.array([[1, 2, 3, 4, 5]]) # .shape: (1, 5)
vec_asnpcol = np.array([[1], [2], [3], [4], [5]]) # shape: (5, 1)


def transpose(v):
    if len(v) == 1: # for row vector; row vectors have 1 row
        dim = v.shape[1]
        col_v = np.zeros((dim, 1))
        for i in range(dim):
            col_v[i, 0] = v[0, i]
        return col_v
    else: # for col vector
        dim = v.shape[0]
        row_v = np.zeros((1, dim))
        for i in range(dim):
            row_v[0, i] = v[i, 0]
        return row_v

    
assert (transpose(vec_asnprow) == vec_asnprow.T).all()
assert (transpose(vec_asnpcol) == vec_asnpcol.T).all()


def transpose_print():
    print("row to col")
    print(vec_asnprow)
    print(transpose(vec_asnprow))
    print("\n\n")
    print("col to row")
    print(vec_asnpcol)
    print(transpose(vec_asnpcol))
# transpose_print()


# basic operations,+-, scalar multiplication, dot product, magnitude, orthogonal projection
# get intuitive understanding of vector addition and subtraction
''' "Do not underestimate the importance of the geometry of vector subtraction: it is the
basis for orthogonal vector decomposition, which in turn is the basis for linear least squares,
which is one of the most important applications of linear algebra in science and engineering." '''

# dot product of orthogonal vectors == 0
# sum of all the elementwise multiplications
def dot(v, w):
    assert v.shape == w.shape
    return np.sum(v * w) 


# dot product with self and then take square root
def mag(v):
    return np.sqrt(dot(v, v))


def decomp(a, b): # a is target vector, b is reference vector
    # orthogonal vector decomposition
    beta = dot(a, b) / dot(b, b) # dot(a, a) if you want to find b - Ba (scaling a)
    x = beta * b
    return (x, a - x)


a = np.random.randn(2)
b = np.random.randn(2)
x, y = decomp(a, b)
orth_test = np.dot(x, y)
assert (a == x + y).all() # sometimes this leads to assertion error
assert orth_test > -0.01 and orth_test < 0.01

# linear weighted combinations, all three have same ans
l1 = 1
l2 = 2
l3 = -3
v1 = np.array([4, 5, 1])
v2 = np.array([-4, 0, -4])
v3 = np.array([1, 3, 2])
print(l1*v1 + l2*v2 + l3*v3)

weights = [l1, l2, l3]
vectors = [v1, v2, v3]

print(sum([l*v for (l,v) in zip(weights, vectors)]))

output = np.zeros(3)
for w, v in zip(weights, vectors):
    output += w*v
print(output)
    
# linear independence
# no vector can be created from linear combinations of other vectors

# ex. 3-3
import matplotlib.pyplot as plt
vec = np.array([1, 3])
xmax = 4
scalars = np.random.uniform(low=-xmax, high=xmax, size=100)
for s in scalars:
    p = s*vec
    print(p)
    plt.plot(p[0], p[1], 'ko')
plt.show()

r3_1 = np.array([3, 5, 1])
r3_2 = np.array([0, 2, 2])

scalars2 = np.random.uniform(low=-xmax, high=xmax, size=100)
for s1, s2 in zip(scalars, scalars2):
    p = s1*r3_1 + s2*r3_2

points = [s1*r3_1 + s2*r3_2 for (s1, s2) in zip(scalars, scalars2)]
xs = [p[0] for p in points]
ys = [p[1] for p in points]
zs = [p[2] for p in points]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(xs, ys, zs)
plt.show()
