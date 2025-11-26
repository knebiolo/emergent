import numpy as np
from emergent.ship_abm.ship_model import ship

# minimal state: x,y,u,v,p,r,phi,psi shapes (8, n)
state0 = np.zeros((8,1))
# pos0 as (2,n)
pos0 = np.array([[0.0],[0.0]])
psi0 = np.array([0.0])
# goal 100m east
goals = np.array([[100.0],[0.0]])

s = ship(state0=state0, pos0=pos0, psi0=psi0, goals_arr=goals)
# tune for clarity
s.lead_time = 10.0
s.dead_reck_sensitivity = 0.25
s.dead_reck_max_corr_deg = 30.0
s.desired_speed = np.array([5.0])

# helper to pretty print
def show_case(curr_vec, label):
    # curr_vec shape (2,n)
    hd, sp = s.compute_desired(goals, x=pos0[0], y=pos0[1], u=0.0, v=0.0, r=0.0, psi=psi0, current_vec=curr_vec)
    print(f"{label}: current={curr_vec.ravel()} -> hd(deg)={np.degrees(hd).ravel()} sp={sp}")

# Case A: no current
curr0 = np.zeros((2,1))
show_case(curr0, 'No current')

# Case B: northward current (pushes ship north while heading east)
curr_north = np.array([[0.0],[0.5]])
show_case(curr_north, 'North current')

# Case C: southward current
curr_south = np.array([[0.0],[-0.5]])
show_case(curr_south, 'South current')

# Case D: strong lateral current exceeding sensitivity
curr_strong = np.array([[0.0],[2.0]])
show_case(curr_strong, 'Strong north current')

# Also test with current_vector passed as 1D
curr1d = np.array([0.0, 0.5])
show_case(curr1d.reshape(2,1), '1D input reshaped')

print('Done')
