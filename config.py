# Extract the real and integer types from the C++ code.
from pathlib import Path
import numpy as np

def to_real_array(val):
    return np.array(val, dtype=float).copy()

def to_integer_array(val):
    return np.array(val, dtype=int).copy()

np_real = np.float64
np_integer = np.int64