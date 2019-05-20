import numpy as np
from matplotlib import pyplot as plt

#Make array
n = int(input('Enter the n number : '))
M = np.random.randn(n,n)
inverseM = np.linalg.inv(M)

#Print Result
a = M @ inverseM
print(a)
