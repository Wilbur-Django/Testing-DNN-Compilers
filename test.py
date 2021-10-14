import os

import numpy as np

a = np.array([1, 3.0, 5.0, 3]).reshape((2, 2))
print(True in np.isnan(a))
