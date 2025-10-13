import numpy as np
from math import cos, sin, sqrt


def lattice_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    '''
    This function computes the lattice matrix from the lattice parameters.

    Args:
    - a (float): Length of lattice vector a.
    - b (float): Length of lattice vector b.
    - c (float): Length of lattice vector c.
    - alpha_deg (float): Angle between lattice vectors b and c in degrees.
    - beta_deg (float): Angle between lattice vectors a and c in degrees.
    - gamma_deg (float): Angle between lattice vectors a and b in degrees.
    '''

    alpha = np.deg2rad(alpha_deg)
    beta  = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)

    ax = a; ay = 0.0; az = 0.0
    bx = b * cos(gamma)
    by = b * sin(gamma)
    bz = 0.0
    cx = c * cos(beta)
    cy = c * (cos(alpha) - cos(beta)*cos(gamma)) / sin(gamma)
    cz = c * sqrt(1 - cos(beta)**2 - ((cos(alpha) - cos(beta)*cos(gamma)) / sin(gamma))**2)

    return np.array([[ax, bx, cx],
                     [ay, by, cy],
                     [az, bz, cz]], dtype=float)


def frac_to_cart(frac, a, b, c, alpha_deg, beta_deg, gamma_deg):
    '''
    Transform fractional coordinates to cartesian coordinates.

    Args:
    - frac (np.ndarray): Fractional coordinates of shape (N, 3).
    - a, b, c (float): Lengths of the lattice vectors.
    - alpha_deg, beta_deg, gamma_deg (float): Angles between the lattice vectors in degrees.
    '''

    lattice_mat = lattice_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg)
    return frac @ lattice_mat.T       # (N,3) * (3,3)^T


def cart_to_frac(cart, a, b, c, alpha_deg, beta_deg, gamma_deg):
    '''
    Transform cartesian coordinates to fractional coordinates.

    Args:
    - cart (np.ndarray): Cartesian coordinates of shape (N, 3).
    - a, b, c (float): Lengths of the lattice vectors.
    - alpha_deg, beta_deg, gamma_deg (float): Angles between the lattice vectors in degrees.
    '''

    lattice_mat = lattice_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg)
    Linv = np.linalg.inv(lattice_mat)
    return cart @ Linv.T