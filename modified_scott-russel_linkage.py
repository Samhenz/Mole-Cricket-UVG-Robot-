# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:23:09 2020

A modified Scott-Russell linkage is presented. It was chosen for replicate 
lateral push technique and trajectory describing the end point of the front legs 
of the Mole Cricket UVG robot.

@author: Hernández, Samuel A.
-based on: https://github.com/pydy/pydy/blob/master/examples/four_bar_linkage/four_bar_linkage_with_motion_constrained_link.ipynb
"""

import sympy as sm
import sympy.physics.mechanics as me
me.init_vprinting()

"""
    Define Variables
"""
# Introduce the generalized coordinates, generalized speeds, and a specified 
# torque to drive one link
q2, q3, = me.dynamicsymbols('q2:4')
u2, u3, = me.dynamicsymbols('u2:4')
T = me.dynamicsymbols('T')

# Introduce all of the necessary constants
l1, l2, l3, l4, l5 = sm.symbols('l1:6')
m1, m2, m3 = sm.symbols('m1:4')
g = sm.symbols('g')
f1 = sm.symbols('f1')

"""
    Reference Frame Orientations
"""
# A reference frame is attached to each link:
# - N for P1 to P3
# - A for P1 to P2
# - B for P4 to P2 to P3
N = me.ReferenceFrame('N')
A = N.orientnew('A', 'Axis', (q2, N.z))
B = A.orientnew('B', 'Axis', (q3, -A.z))

C = B.orientnew('C','Axis',((sm.pi/2, B.z)))
"""
    Point Locations
"""
P1 = me.Point('P1')
P2 = P1.locatenew('P2', l1*A.x)
P3 = P2.locatenew('P3', l2*B.x)
P4 = P2.locatenew('P4', -l3*B.x)

P5 = P4.locatenew('P5', l4*C.x)
P6 = P4.locatenew('P6', -l5*C.x)

# Mass centers locations
crank_mc = P1.locatenew('crank_mc',0.5*l1*A.x)
rod_mc = P2.locatenew('rod_mc',0.5*(l2+l3)*B.x)

efector_mc = P4.locatenew('efector_mc',0.5*(l4+l5)*C.x)


"""
    Configuration Constraints
"""
# This is the vector equation that will be used to create the configuration 
# constraints
loop = P3.pos_from(P1) - (l1*sm.cos(q2)+l2*sm.cos((q3-(sm.pi/2))))*N.x

# There are two configuration constraints but only one of the configuration 
# variables is necessary for speciying the complete coniguration of the system
config_con1 = loop.dot(N.x).simplify()
config_con2 = loop.dot(N.y).simplify()

"""
    Kinematical Differential Equations
"""
# Introduce a generalized speed for each generalized coordinate (for both 
# independent and dependent generalized coordinates)
qdots = {q.diff(): u for q, u in zip((q2, q3), (u2, u3))}

"""
    Set Velocities in Terms of The Generalized Speeds
"""
A.set_ang_vel(N, u2*N.z)
B.set_ang_vel(A, -u3*A.z)
C.set_ang_vel(A, -u3*A.z)

P1.set_vel(N, 0)
P2.v2pt_theory(P1, N, A)
P3.v2pt_theory(P2, N, B)
P4.v2pt_theory(P2, N, B)

P5.v2pt_theory(P4, N, C)
P6.v2pt_theory(P4, N, C)

# Velocities of mass centers
crank_mc.v2pt_theory(P1, N, A)
rod_mc.v2pt_theory(P2, N, B)
efector_mc.v2pt_theory(P4, N, B)

"""
    Motion Constraints
"""
# P3 can only have motion along the line from P3 to P1 (slider)
# This is a single true non-holonomic constraint
mot_con = P3.vel(N).dot(N.y).simplify()

t = me.dynamicsymbols._t

"""
    Accelerations
"""
# Make sure all of the accelerations relevant to forming Kane's equations 
# do not have any  𝑞_dots.
A.ang_acc_in(N)

P2.acc(N)
P3.acc(N)
#P4.acc(N)

P5.acc(N)
P6.acc(N)

crank_mc.acc(N)
rod_mc.acc(N)
efector_mc.acc(N)

"""
    Rigid Bodies
"""
# Create the necessary bodies
IA = m1*l1*l1/12
IB = m2*(l2+l3)*(l2+l3)/12
IC = m3*(l4+l5)*(l4+l5)/12

A_inertia=(me.inertia(A,0,IA,IA),crank_mc)
B_inertia=(me.inertia(B,0,IB,IB),rod_mc)
C_inertia=(me.inertia(C,0,IC,IC),efector_mc)

crank = me.RigidBody('A', crank_mc, A, m1, A_inertia)
rod = me.RigidBody('B', rod_mc, B, m2, B_inertia)
efector = me.RigidBody('C', efector_mc, C, m3, C_inertia)

particle = me.Particle('P4', P4, m3)

bodies = [crank, rod, particle]

"""
    Loads
"""
# A specified torque is applied between A and N. Also you can apply something like...
#loads = [(P2, -m2*g*N.y),
#         (P3, -m3*g*N.y),
#         (A, T*N.z)]
loads = [(P3, -f1*N.x),(A, T*N.z)]

"""
    Equations of Motion
"""
# Solve the dynamics
kane = me.KanesMethod(N, # inertial reference frame
                      [q2], # independent generalized coordinates
                      [u2], # independent generalized speed
                      kd_eqs=[qd - u for qd, u in qdots.items()], # q' = u for all coordinates
                      q_dependent=[q3], # depdendent coordinates from the kinematic loop
                      configuration_constraints=[config_con2],  # kinematic loop config constraints
                      u_dependent=[u3], # dependent generalized speeds
                      velocity_constraints=[mot_con],  # nonholonomic motion constraints
                      # acc constraints are required to ensure all qdots are properly substituted
                      acceleration_constraints=[mot_con.diff(t).subs(qdots)])

fr, frstar = kane.kanes_equations(bodies, loads=loads)
zero = fr + frstar


me.find_dynamicsymbols(zero)

# Note the order of the generalized coordinates and speeds
kane.q
kane.u

# Find the equations of motion

# All  𝑞_ddots have been eliminated from the equations of motion
me.find_dynamicsymbols(kane.mass_matrix_full)
me.find_dynamicsymbols(kane.forcing_full)

# The mass matrix and the forcing function can be taken out of the
# KanesMethod object.
MM = kane.mass_matrix
forcing = kane.forcing
kdd = kane.kindiffdict()

# And those can be used to find the equations of motion.
qudots = kane.rhs()
qudots = qudots.subs(kdd)
qudots.simplify()
print(qudots)

"""
    Integrate the Equations of Motion
"""

import numpy as np
from scipy.optimize import fsolve
from pydy.system import System

sys = System(kane)

l1_val = 2.0
l2_val = 3.0
l3_val = 4.5
f1_val = 3.0
l4_val = 0.15
l5_val = 0.15

sys.constants = {l1: l1_val,
                 l2: l2_val,
                 l3: l3_val,
                 f1: f1_val,
                 #l4: l4_val,
                 #l5: l5_val,
                 #g: 9.81,
                 m1: 2.0,
                 m2: 3.0,
                 m3: 1.0}

"""
    Initial Conditions
"""
# These must satisfy the configuration constraints so use "fsolve" to find a 
# satifactory numerical solution

# Choose a value for the independent generalized coordinate:
q2_0 = np.deg2rad(10.0)

# Create a function that evaluates the constraints and find compatible values 
# for the dependent generalized coordinates.
eval_config_con = sm.lambdify((q2, q3, l1, l2),
                              sm.Matrix([config_con2]))

eval_config_con_fsolve = lambda x, q2, l1, l2: np.squeeze(eval_config_con(q2, x[0], l1, l2))

q3_0 = fsolve(eval_config_con_fsolve, np.ones(1), args=(q2_0, l1_val, l2_val))

sys.initial_conditions = {q2: q2_0,
                          q3: q3_0,
                          u2: 0.0,
                          u3: 0.0}

"""
    Set Output Times
"""
duration = 6.5
fps = 100.0
sys.times = np.linspace(0.0, duration, num=int(fps*duration))

"""
    Specified Torque
"""
# Create a function that turns the torque on for a "t" duration and then off
# This will get the linkage moving.
def step_pulse(x, t):
    if t < 6.2:
        T = 10.0
    else:
        T = 0.0
    return np.array([T])

sys.specifieds = {T: step_pulse}

# Integrate
x = sys.integrate()

"""
    Plot the State Trajectories
"""
import matplotlib.pyplot as plt

# =============================================================================
# Create a plot for the State Trajectories
# =============================================================================
"""
fig, axes = plt.subplots(2, 2, sharex=True)
fig.set_size_inches(10, 10)

for i, (xi, ax, s) in enumerate(zip(x.T, axes.T.flatten(), sys.states)):
    ax.plot(sys.times, np.rad2deg(xi))
    title = sm.latex(s, mode='inline')
    ax.set_title(title)
    if 'q' in title:
        ax.set_ylabel('Angle [deg]')
    else:
        ax.set_ylabel('Angular Rate [deg/s]')

axes[1, 0].set_xlabel('Time [s]')
axes[1, 1].set_xlabel('Time [s]')

plt.tight_layout()
"""

"""
#    Inspect the Configuration and Motion Constraints
"""
# =============================================================================
# Create a plot for the Motion constraints
# =============================================================================
#motion_cons = 0 # 1/0

config_constraint_vals = eval_config_con(x[:, 0],  # q2
                                         x[:, 1],  # q3
                                         l1_val, l2_val).squeeze()
"""
fig, ax = plt.subplots()
ax.plot(sys.times, config_constraint_vals.T)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Configuration Constraint Value [m]')
"""
eval_motion_con = sm.lambdify((q2, q3, u2, u3, l1, l2, l3),
                              sm.Matrix([mot_con]))

motion_constraint_vals = eval_motion_con(x[:, 0],  # q2
                                         x[:, 1],  # q3
                                         x[:, 2],  # u2
                                         x[:, 3],  # u3
                                         l1_val, l2_val, l3).squeeze()

# These constraints grow with time and can exceed a desirable bound, so be aware!
"""
fig, ax = plt.subplots()
ax.plot(sys.times, motion_constraint_vals.T)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Motion Constraint Value [m/s]')
ax.legend(['Nonholonomic'])
"""

"""
    Animate the Linkage
"""
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# The location of the moving points will be needed
P2.pos_from(P1).express(N).simplify()
P3.pos_from(P2).express(N).simplify()
P4.pos_from(P2).express(N).simplify()

P5.pos_from(P4).express(N).simplify()
P6.pos_from(P4).express(N).simplify()


q2_vals = x[:, 0]
q3_vals = x[:, 1]

p2_xy = np.array([l1_val*np.cos(q2_vals),
                  l1_val*np.sin(q2_vals)])

p3_xy = p2_xy + np.array([l2_val*np.cos(q2_vals - q3_vals),
                          l2_val*np.sin(q2_vals - q3_vals)])

p4_xy = p2_xy + np.array([-l3_val*np.cos(q2_vals - q3_vals),
                          -l3_val*np.sin(q2_vals - q3_vals)])
    
p5_xy = p4_xy + np.array([-l4_val*np.sin(q2_vals - q3_vals),
                           l4_val*np.cos(q2_vals - q3_vals)])
    
p6_xy = p4_xy + np.array([l5_val*np.sin(q2_vals - q3_vals),
                         -l5_val*np.cos(q2_vals - q3_vals)])
    
# =============================================================================
# Create a base plot with the initial state values
# =============================================================================

fig, ax = plt.subplots()
fig.set_size_inches((12, 14))
line, = ax.plot([0.0, p2_xy[0, 0], p3_xy[0, 0], p4_xy[0, 0], p5_xy[0, 0], p6_xy[0, 0]],
                [0.0, p2_xy[1, 0], p3_xy[1, 0], p4_xy[1, 0], p5_xy[1, 0], p6_xy[1, 0]])
title = 'Time = {:0.1f} seconds'
ax.set_title(title.format(0.0))
ax.set_ylim((-6.0, 6.0))
ax.set_xlim((-8.0, 6.0))
ax.set_aspect('equal')


# =============================================================================
# Create a function for animate the linkage
# =============================================================================
def update(i):
    xdata = [0.0, p2_xy[0, i], p3_xy[0, i], p4_xy[0, i], p5_xy[0, i], p6_xy[0, i]]
    ydata = [0.0, p2_xy[1, i], p3_xy[1, i], p4_xy[1, i], p5_xy[1, i], p6_xy[1, i]]
    line.set_data(xdata, ydata)
    ax.set_title(title.format(sys.times[i]))
    return line,

ani = FuncAnimation(fig, update, save_count=len(sys.times))
HTML(ani.to_jshtml(fps=fps))

# =============================================================================
# Create a plot with the end point values
# =============================================================================
"""
fe_xval = p4_xy[0,:]
fe_yval = p4_xy[1,:]

fig3, ax3 = plt.subplots()
fig3.set_size_inches((12, 14))
ax3.plot(fe_xval, fe_yval)
ax3.set_title('End Point Position Trajectory')
ax3.set_ylim((-6.0, 6.0))
ax3.set_xlim((-8.0, 6.0))
ax3.set_aspect('equal')
"""


