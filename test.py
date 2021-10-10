import os

import numpy as np

from compile.xla.compile_manager import DefaultManager

temp_dir = "/export/temp"

a = np.load("/root/data/data.npy")

m = DefaultManager()

lib = np.ctypeslib.load_library('libmodel', temp_dir)
print(m.predict(lib, a))
