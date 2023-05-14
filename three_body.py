import numpy as np
import matplotlib.pyplot as ply
import numba 
import pandas as pd

# reading initialization from csv files
# and convert to numpy vectors
df_comp = pd.read_csv('comp.csv')
df_center = pd.read_csv('mass.csv')

def accel_majorStars(pos: np.ndarray, G : float, M : float):
    n = 2 # number of major mass body
    A_pos = pos[0]
    B_pos = pos[1]
    a = np.zeros(n, 3) # acceleration array

    r_vec = A_pos - B_pos 
    r_mag = np.linalg.norm(r_vec) 
    acc = -(G*M*r_vec/r_mag**3)
    A_acc = acc # accel of mass body A
    B_acc = -acc # accel of mass body B

    a[0] = A_acc 
    a[1] = B_acc
    return a 

def accel_satellites(pos_satellites: np.ndarray, pos_majorStars: np.ndarray, G: float, M: float):
    n = pos_majorStars[:,0].size # number of major masses
    m = pos_satellites[:,0].size # number of satellites
    a = np.zeros(m, 3)

    for i in range(n):
        for j in range(m):
            r_vec = pos_satellites[j] - pos_majorStars[i] 
            r_mag = np.linalg.norm(r_vec) 
            a[j] = a[j] - (G*M*r_vec/r_mag**3)
    
    return a

def leapfrog(r_star: np.ndarray, r_satellites: np.ndarray, dt: float, G: float, M: float):

    # major stars leapfrog
    v_star = v_star+ 0.5*dt*accel_majorStars(r_star, G, M)
    r_star = r_star+ v_star*dt
    v_star = v_star+ 0.5*dt*accel_majorStars(r_star, G, M)

    # satellites leapfrog
    v_satellites = v_satellites + 0.5*dt*accel_satellites(r_satellites, r_star, G, M)
    r_satellites = r_satellites + v_satellites*dt
    v_satellites = v_satellites + 0.5*dt*accel_satellites(r_satellites, r_star, G, M)

    return r_star, v_star, r_satellites, v_satellites

