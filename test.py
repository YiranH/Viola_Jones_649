import numpy as np
from haar import *

if __name__ == "__main__":
    a = np.array([
    [5, 2, 3, 4, 1],
    [1, 5, 4, 2, 3],
    [2, 2, 1, 3, 4],
    [3, 5, 6, 4, 5],
    [4, 1, 3, 2, 6]])
    a_ii = get_integral_image(a)

    print(a_ii)

    c = HaarFeature(1,(1,1),1,1,1,1)

    val = c.get_sum(a_ii, (3,3), (4,4))
    print(val)