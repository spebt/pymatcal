import numpy as np
m = np.array([[1,2,3],[1,1,1],[1,1,1]])
a = np.array([[1,1,1],[2,2,2]])
print(np.matmul(a[0],m.T)-np.array([1,1,1]))
print('Loop:')
for row in a:
    print(np.matmul(m,row)-np.array([1,1,1]))
# conclusion, numpy array can be treated as row array for easy logic