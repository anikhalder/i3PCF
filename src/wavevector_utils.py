import numpy as np

######################################################################################

## 2D wavevector (l) arithmetic utility functions

## m stands for minus (-) and p stands for plus (+)

def l_ApB(l_A, phi_A, l_B, phi_B):
    val = np.array( np.sqrt(l_A*l_A + l_B*l_B + 2*l_A*l_B*np.cos(phi_A-phi_B)) )
    val[np.isnan(val)] = 0.0
    return val

def phi_ApB(l_A, phi_A, l_B, phi_B):
    x = l_A*np.cos(phi_A) + l_B*np.cos(phi_B)
    y = l_A*np.sin(phi_A) + l_B*np.sin(phi_B)
    phi = np.array( np.arctan2(y,x) )
    mask = phi < 0
    phi[mask] = phi[mask] + 2*np.pi
    return phi

def l_ApBpC(l_A, phi_A, l_B, phi_B, l_C, phi_C):
    val = np.array(np.sqrt(l_A*l_A + l_B*l_B + l_C*l_C + 2.*l_A*l_B*np.cos(phi_A-phi_B) + 2.*l_A*l_C*np.cos(phi_A-phi_C) + 2.*l_B*l_C*np.cos(phi_B-phi_C)))
    val[np.isnan(val)] = 0.0
    return val

def phi_ApBpC(l_A, phi_A, l_B, phi_B, l_C, phi_C):
    l_sum = l_ApB(l_A, phi_A, l_B, phi_B)
    phi_sum = phi_ApB(l_A, phi_A, l_B, phi_B)
    return phi_ApB(l_sum, phi_sum, l_C, phi_C)

######################################################################################

## 3D wavevector (q) arithmetic utility functions

def spherical_to_cartesian_3D(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def vector_3D_length(x, y, z):
    return np.array( np.sqrt(x*x + y*y + z*z) )

def cartesian_to_spherical_3D(x, y, z):
    r = vector_3D_length(x, y, z)
    theta = np.array( np.arctan2(np.sqrt(x*x+y*y), z) )
    phi = np.array( np.arctan2(y,x) )
    mask = phi < 0
    phi[mask] = phi[mask] + 2*np.pi
    return r, theta, phi