import numpy as np
import matplotlib.pyplot as plt
import numba 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# unit system
# 

def accel_majorStars(pos: np.ndarray, G : float, M : float):
    n = 2 # number of major mass body
    A_pos = pos[0]
    B_pos = pos[1]
    a = np.zeros((n, 3)) # acceleration array

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
    a = np.zeros((m, 3))

    for i in range(n):
        for j in range(m):
            r_vec = pos_satellites[j] - pos_majorStars[i] 
            r_mag = np.linalg.norm(r_vec) 
            a[j] = a[j] - (G*M*r_vec/r_mag**3)
    
    return a

def leapfrog(r_star: np.ndarray, v_star: np.ndarray, r_satellites: np.ndarray, v_satellites: np.ndarray, dt: float, G: float, M: float):
    n = r_satellites[:,0].size

    # major stars leapfrog
    v_star = v_star+ 0.5*dt*accel_majorStars(r_star, G, M)
    r_star = r_star+ v_star*dt
    v_star = v_star+ 0.5*dt*accel_majorStars(r_star, G, M)

    # satellites leapfrog
    v_satellites = v_satellites + 0.5*dt*accel_satellites(r_satellites, r_star, G, M)
    r_satellites = r_satellites + v_satellites*dt
    v_satellites = v_satellites + 0.5*dt*accel_satellites(r_satellites, r_star, G, M)

    return r_star, v_star,r_satellites, v_satellites 



# Constants
G = 4.498*10**(-6) # kpc^3/(M_solar * Earth_year^2)
M = 1 # 1 M_solar
dt = 0.01 # 1 earth_year
T = 1 # total time in earth_year
step = int(T/dt) # total number of steps, should be INT
L_scale = 1 # kpc
v_scale = 1 * (1/(3.08*10**31)) * (3.15*10**7/1)**2 # from cm/s^2 to kpc/year^2

# reading initialization from csv files
# and convert to numpy vectors
df_comp = pd.read_csv('comp.csv')
df_center = pd.read_csv('mass.csv')

# output container (currently using python.List)
out_star = [] 
out_satellite = []

# converting position and velocity to np.narray
position_star = df_center[['X', 'Y', 'Z']].to_numpy()
velocity_star = df_center[['V_x', 'V_y', 'V_z']].to_numpy()
position_satellite = df_comp[['X', 'Y', 'Z']].to_numpy()
velocity_satellite= df_comp[['V_x', 'V_y', 'V_z']].to_numpy()

# store initial
out_star.append(position_star)
out_satellite.append(position_satellite)

# loop
for i in range(int(step)):
    position_star, velocity_star, position_satellite, velocity_satellite = leapfrog(position_star, velocity_star, position_satellite, velocity_satellite, dt = dt, G = G, M = M)
    out_star.append(position_star)
    out_satellite.append(position_satellite)

out_star = np.array(out_star)
out_satellite = np.array(out_satellite)

star = out_star.reshape((step+1)*2,3)



# Create the figure and axes for the plot
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')


x = star[:,0]
print(x)
y = star[:,1]
z = star[:,2]
ax.view_init(90, -90, 0)
# Create the initial point in the plot
point, = ax.plot(x[0], y[0], z[0], 'bo')

# Define the update function for the animation
def update(frame):
    # Update the position of the point
    point.set_data(x[frame:frame+2], y[frame:frame+2])
    point.set_3d_properties(z[frame:frame+2])
    return point,

# Create the animation object
anim = FuncAnimation(fig, update, frames = len(x), blit=True, interval = 50)

ax.set_xlabel("x (kpc)")
ax.set_ylabel("y (kpc)")
ax.set_zlabel("z (kpc)")
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(-50, 50)
ax.set_title('Mice Collision')
# Show the plot
plt.legend()
plt.show()