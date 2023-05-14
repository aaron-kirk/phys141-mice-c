import numpy as np
import matplotlib.pyplot as ply
import numba 
import pandas as pd

# reading initialization from csv files
# and convert to numpy vectors
df_comp = pd.read_csv('comp.csv')
df_center = pd.read_csv('mass.csv')
Center = df_center.to_numpy()
print(Center)
