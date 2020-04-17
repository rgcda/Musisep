#!python3

"""
Helper script to extract the performance numbers from running on
Jaiswal's data.
"""

import ast
import re
import sys
import numpy as np

values = sys.stdin.read()
values = re.sub(r'([^\[])\s+([^\]])', r'\1,\2', values)
values = ast.literal_eval(values)
values = np.mean(values, axis=2)

n = len(values)
np.savetxt(sys.stdout.buffer,
           np.vstack((np.arange(1, n+1), values.T)).T,
           delimiter=' ')

print(np.mean(values, axis=0))
