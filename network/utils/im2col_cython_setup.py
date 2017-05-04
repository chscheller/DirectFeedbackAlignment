from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
  name='im2col',
  ext_modules=cythonize("im2col_cython.pyx"),
  include_dirs=[np.get_include()]
)