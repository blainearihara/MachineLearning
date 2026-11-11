# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:15:07 2026

@author: blain
"""

import numpy as np
import torch

#%% ARRAY CREATION
# FROM A LIST / LITERAL VALUES
# row vector
a = np.array([1, 2, 3, 4]);a
    
# column vector
b = np.array([[1],[2],[3],[4]]);b

# 2×2 matrix
M = np.array([[1,2],[3,4]]);M

# RANGES — LINSPACE / ARANGE
# (start, stop, number of elements)
a = np.linspace(0, 10, 11);a

# (start, stop, step)
b = np.arange(0, 11, 2);b

# ZEROS, ONES, IDENTITY, RANDOM
a = np.zeros((3,4));a # shape tuple, not two args
b = np.ones((2,2));b
c = np.eye(3);c
d = np.random.rand(2,3);d  # uniform [0,1)
e = np.random.randn(2,3);e  # standard normal

# FULL, EMPTY, LIKE (np.empty is faster than zeros when you'll overwrite every element anyway)
a = np.full((3,3), 7);a      # fill with value
b = np.empty((3,3));b        # uninitialized (fast)
c = np.zeros_like(M);c       # same shape/dtype as M
d = np.ones_like(M);d


#%% SHAPE AND RESHAPE
# 2×3 matrix
A = np.array([[1,2,3],[4,5,6]]);A
# INSPECTING SHAPE & SIZE
B = A.shape;B       # tuple e.g. (3,4)
B = A.shape[0];B   # number of rows
B = A.shape[1];B   # number of cols
B = A.size;B        # total elements
B = A.ndim;B        # number of dimensions

# FLATTEN AND RAVEL
# returns a new copy of the data as a 1-D array Safe to modify without affecting A. Slightly slower because memory is always allocated.
B = A.flatten();B
B = A.flatten(order='F');B
# returns a 1-D array but tries to return a view first — meaning it points to the same memory as A. Only falls back to a copy when the array's memory layout makes a view impossible. Faster when it works as a view, but you have to be careful: modifying the result may modify A
B = A.ravel();B       # view if possible (faster)
B = A.ravel(order='F');B  # column-major like MATLAB A(:)
# A.view() is lower-level and more general. The result always shares memory with A. You're telling NumPy "look at the same bytes, but interpret them differently."
# in NumPy: It reinterprets the same memory block with a different dtype.
# in PyTorch: It reinterprets the same memory block with a different shape or dtype.
x = torch.arange(6);x
z = x.view(2,3);z

# RESHAPE
A = np.arange(0,6);A
B = A.reshape(2, 3);B
B = A.reshape(-1, 3);B   # -1 means "figure it out"
B = A.reshape(2, 3, order='F');B  # column-major like MATLAB

# TRANSPOSE AND SQUEEZE
B = A.reshape(1,-1);B # shape from (6,) to (1,6)
B = A.reshape(-1,1);B # shape from (6,) to (6,1)
B = B.T             # transpose (no conjugate) shape (6,1)
B = A.conj().T     # conjugate transpose
B = np.squeeze(A)  # or A.squeeze() shape (6,)

#%% MANIPULATION
A = np.array([[1,2,3],[4,5,6]]);A
B = np.array([[7,8,9]]);B
C = np.array([[7,8],[9,10]]);C
# CONCATENATE (HORZCAT / VERTCAT)
D = np.hstack([A, C]);D          # horizontal
D = np.concatenate([A,C], axis=1);D  # along axis 1

D = np.vstack([A, B]);D          # vertical
# before using concatenate in vertical, reshape B from (3,) to (1,3)
B = B.reshape(1,-1)
D = np.concatenate([A,B], axis=0);D  # along axis 0

# TILE / REPEAT
v = np.arange(0,4);v
D = np.tile(A, (3, 2));D     # tile 2 rows × 3 cols ... repmat()
D = np.repeat(v, 3);D   # repeat each element 3× ... repelem()
D = np.repeat(v, [1,2,3]);D   # different count each

# SORT
v = np.array([4, 1, 2, 0, 3, 5]);v
A = np.array([[5,8,6], [9,2,4], [3,7,1]]);A
D = np.sort(v);D              # sorted copy
D = np.sort(A, axis=0);D     # sort each column
D = np.sort(A, axis=1);D     # sort each row
idx = np.argsort(v);idx    # indices only
D = v[np.argsort(v)];D       # same as np.sort(v)

# UNIQUE, FLIP, ROLL
D = np.unique(v);D
D = np.unique(A, axis=0);D     # unique rows
D = np.flip(A, axis=1);D       # fliplr
D = np.flip(A, axis=0);D       # flipud
k = 1
D = np.roll(A, k, axis=1);D    # circshift(A, k, dim)
D = np.roll(A, k, axis=0);D    # circshift(A, k, dim)


#%% INDEXING
# BASIC INDEXING — 0-BASED!
A = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]]);A
b = A[1, 2];b         # row 2, col 3 (0-based!)
b = A[-1, -1];b      # last element — negative indexing!

# SLICING — EXCLUSIVE STOP
b = A[:, 1:3];b      # (stop exclusive!)
b = A[:, 1];b        # all rows, col index 1
b = A[:, ::2];b      # every other row

# LIST INDEXING
b = A[:, [0,2,4]];b       # rows 0,2,4
b = A[[0,1], [0,5]];b    # elements (0,1) and (2,3)

# LINEAR INDEXING (FIND)
b = A.flat[4];b             # 4th element, row-major
row_ind, col_ind = np.where(A > 10);
row_ind, col_ind
b = A[row_ind,col_ind];b  # or: A[A > 3]

#%% MASKS AND WHERE
A = np.array([[-1,-2,-3,-4,-5,-6],[7,8,9,10,11,12]]);A
mask = A > 3            # boolean array, same shape
mask = (A > 1) & (A < 5)   # & not 'and'!
mask = np.isnan(A)         # NaN mask
mask = ~mask               # invert (NOT)

# APPLYING A MASK (LOGICAL INDEXING)
A = np.array([[-1,-2,-3,-4,-5,-6],[7,8,9,10,11,12]]);A
mask = (A == 9) | (A == 4)
B = A[mask];B             # extract as 1-D array
A[mask] = 0;A        # set matching elements
inds = np.where(mask);inds

B = np.where(A > 0, A, 0);B  # ReLU — clip negatives
B = A.copy()
B[B < 0] = 0;B

A[np.isnan(A)] = 0    # replace NaNs



#%% BROADCASTING
'''
The single rule that covers all cases: align shapes from the right, then check each dimension pair.
Two dimensions are compatible if they are equal, or if one of them is 1.
A missing dimension on the shorter array is treated as 1.
(3,)   →  treated as  (1, 3)
(2, 1) →  stretches to (2, 3) when paired with (2, 3)
'''
# EXAMPLE 1: Scalar + array — the simplest case
# The scalar 10 is "stretched" to match a's shape.
a = np.array([1, 2, 3, 4]);a
b = a + 10;b  # [11, 12, 13, 14]
# Equivalent to: a + np.array([10, 10, 10, 10])

# EXAMPLE 2: 1-D array + 1-D array (same shape)
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
c = a + b;c  # [11, 22, 33]

# EXAMPLE 3: 2-D array + 1-D array (add to each row)
A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2,3)
b = np.array([10, 20, 30]) # shape (3,)
C = A + b;C
# [[11, 22, 33],
#  [14, 25, 36]]

# EXAMPLE 4: 2-D array + column vector (add to each column)
A = np.array([[1, 2, 3],
              [4, 5, 6]])        # shape (2,3)
b = np.array([[10],
              [20]])             # shape (2,1)
C = A + b;C
# [[11, 12, 13],
#  [24, 25, 26]]

# EXAMPLE 5: Outer product — two 1-D arrays broadcast against each other
rows = np.array([0, 10, 20])    # shape (3,)
cols = np.array([1, 2, 3, 4])  # shape (4,)

# Reshape rows to a column vector:
result = rows[:, np.newaxis] + cols
# shape (3,4)
# [[ 1,  2,  3,  4],
#  [11, 12, 13, 14],
#  [21, 22, 23, 24]]