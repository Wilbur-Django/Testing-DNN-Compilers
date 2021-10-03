import os

import numpy as np

temp_dir = "/export/d2/dwxiao/temp"

a = np.random.normal(size=(4, 32, 32, 64)).astype(np.float32)

np.savetxt(os.path.join(temp_dir, "a.txt"), a.flatten(), fmt="%.8f")

