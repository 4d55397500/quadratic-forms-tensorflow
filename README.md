# quadratic-forms-tensorflow


Quadratic forms are expressions of the form `x_i M_ik x_k` (implicit summation over indices)
and appear commonly in SVM kernels and loss functions. This wiki presents techniques for creating and manipulating them in Tensorflow.

Alphabetic indices `a,b,c,..` will denote samples, and indices `i,j,k..` will refer to dimensions in the vector space. So, for example, a tensor `x` of shape (batch size, dimension) can be denoted `x_ai`.

Quadratic forms in Tensorflow are then expressions of form `A_ab M_ik x_ai x_bi`. These objects are scalars, and appear commonly as say loss functions. Intermediate contractions can also be meaningful; defining the rank-4 tensor `T_abij := x_ai x_bj` the game is contractions over `T_abij`. Usually such contractions are in diagonal form (symmetric forms can always be diagonalized) but this is not necessary. A common example of such an expression in diagonal form is the pairwise squared difference sum `sum_(a,b) { (x_a - x_b)^2}`; this expression clearly preserves `SO(N)` rotational invariance.

#### Basic quadratic forms

To begin, consider common quadratic forms over `x_ai`. The dot product between two samples is `x_ai x_ib`.
In Tensorflow this is the matrix product of a matrix with its transpose (in all examples below `x = x_ai`, a rank-2 tensor of shape (batch size, dimension)):
```
def dot_product(x):
    return tf.matmul(x, tf.transpose(x))
```

Therefore the norm squared of a given sample is simply a particular diagonal component of the dot product matrix `x_ai x_ib`, that is `x_a^2 = x_ai x_ia` (no summation over `a`). This can also be expressed in Tensorflow:

```
def norm_squares(x):
    return tf.reduce_sum(tf.square(x), axis=1) 
```

#### Pairwise squared difference sum

Consider the sum over all pairwise square differences between samples: `sum_(a,b) { (x_a - x_b)^2 }`. This can be expanded out to `sum_(a,b) { (x_a^2 - 2 x_a x_b + x_b^2) } = 2 * N sum_(a) { x_a^2 } - 2 * sum_(a,b) { x_a x_b }`
Using the above expressions this becomes:

```
def pairwise_squared_diff_sum(x):
    N = tf.shape(x)[0]
    first_term = 2 * N * tf.reduce_sum(norm_squares(x))
    second_term = -2 * tf.reduce_sum(dot_product(x))
    return first_term + second_term
```

#### Kernels
A kernel `K(x_a, x_b)` is a quadratic form or a function of a quadratic form. The tensor `k_ab := K(x_a, x_b)` often needs to be computed.
One example is the Gaussian RBF kernel `k_ab = exp(-gamma * (x_a - x_b)^2)` Again the squared difference is expanded, and we make use of the broadcasting property of tf.subtract (smaller arrays are automatically resized to match the larger array in the subtraction):

```
def gaussian_rbf_kernel(x):
    # x_a^2
    norm_squares = tf.reduce_sum(tf.square(x), axis=1) 
    # q_ab = (x_a - x_b)^2 = x_a^2 I_b - 2 x_a x_b + I_a x_b^2, where I_a is a vector of all 1s.
    q = tf.add(tf.subtract(norm_squares, 2 * tf.matmul(x, tf.transpose(x)))), tf.transpose(norm_squares))
    gamma = 50
    # k_ab = exp(-gamma * q_ab)
    k = tf.exp(-gamma * q)
    return k
```
