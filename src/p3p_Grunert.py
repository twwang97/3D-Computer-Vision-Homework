############################
#                          #
# P3P: Grunert method      #
#                          #
# Author: David Wang       #
# Created on Oct. 20, 2022 #
#                          #
############################

import numpy as np

def findLengths(cos, dist):

    K1, K2 = (dist[2] /dist[1])**2, (dist[2]/dist[0])**2
    lengths = []
    a = [0., 0., 0., 0., 0.]
    # polynomial: a0 * x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4
    a[0] = (K1 * K2 - K1 - K2)**2 - 4. * K1 * K2 * (cos[2]**2)
    if a[0] == 0:
        return False
    a[1] = 4. * (K1 * K2 - K1 - K2) * K2 * (1. - K1) * cos[0] \
            + 4. * K1 * cos[2] * ((K1 * K2 - K1 + K2) * cos[1] + 2. * K2 * cos[0] * cos[2])
    a[2] = (2. * K2 * (1. - K1) * cos[0])**2 \
            + 2. * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) \
            + 4. * K1 * ((K1 - K2) * (cos[2]**2) + K1 * (1. - K2) * (cos[1]**2) - 2. * (1. + K1) * K2 * cos[0] * cos[1] * cos[2])
    a[3] = 4. * (K1 * K2 + K1 - K2) * K2 * (1. - K1) * cos[0] \
            + 4. * K1 * ((K1 * K2 - K1 + K2) * cos[1] * cos[2] + 2. * K1 * K2 * cos[0] * (cos[1]**2))
    a[4] = (K1 * K2 + K1 - K2)**2 - 4. * (K1**2) * K2 * (cos[1]**2)
    
    # at most 4 real-valued solutions to (a0 * x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4)
    x_set = np.roots(a)
    
    for x in x_set:
        if np.isreal(x) == False:
            continue
        x = np.real(x)
        a = np.sqrt((dist[0]**2) / (1 + (x)**2 - 2*x*cos[0]))
        b = x * a
        m, m_prime = 1-K1, 1
        p, p_prime = 2 * (K1*cos[1] - x*cos[2]), 2 * (-x*cos[2])
        q, q_prime = x**2 - K1, (x**2)*(1-K2) + 2*x*K2*cos[0] - K2
        y = (m*q_prime - m_prime*q) / (p*m_prime - p_prime*m)
        c = y * a
        lengths.append([a, b, c]) # solution to lengths
    return lengths

def cosine_angle(v0, v1):
	return np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))

def trilateration(lengths, x_random):
	l1 = lengths[0]
	l2 = lengths[1]
	l3 = lengths[2]
	p1 = x_random[0]
	p2 = x_random[1]
	p3 = x_random[2]
	e_x = (p2 - p1) / np.linalg.norm(p2 - p1)
	i = np.dot(e_x, (p3 - p1))
	e_y = (p3 - p1 - (i * e_x)) / (np.linalg.norm(p3 - p1 - (i * e_x)))
	e_z = np.cross(e_x,e_y)
	d = np.linalg.norm(p2 - p1)
	j = np.dot(e_y, (p3 - p1))
	x = ((l1**2) - (l2**2) + (d**2) ) / (2. * d)
	y = (((l1**2) - (l3**2) + (i**2) + (j**2)) / (2. * j)) - ((i / j) * (x))
	z1 = np.sqrt(l1**2 - x**2 - y**2)
	z2 = np.sqrt(l1**2 - x**2 - y**2) * (-1)
	trans_1 = p1 + (x * e_x) + (y * e_y) + (z1 * e_z)
	trans_2 = p1 + (x * e_x) + (y * e_y) + (z2 * e_z)
	return [trans_1, trans_2]


def rot2vec(v0, v1):
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    # The rotation axis
    axis = np.cross(v0, v1)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len
    x = axis[0]
    y = axis[1]
    z = axis[2]

    cos1 = cosine_angle(v0, v1)
    sin1 = np.sqrt(1 - cos1**2)
    unit1 = 1.0
    R = np.zeros((3, 3)) # rotation matrix
    R[0, 0] = unit1 + (unit1 - cos1) * (x**2 - unit1)
    R[0, 1] = -z * sin1 + (unit1 - cos1) * x * y
    R[0, 2] = y * sin1 + (unit1 - cos1) * x * z
    R[1, 0] = z * sin1 + (unit1 - cos1) * x * y
    R[1, 1] = unit1 + (unit1 - cos1) * (y**2 - unit1)
    R[1, 2] = -x * sin1 + (unit1 - cos1) * y * z
    R[2, 0] = -y * sin1 + (unit1 - cos1) * x * z
    R[2, 1] = x * sin1 + (unit1 - cos1) * y * z
    R[2, 2] = unit1 + (unit1 - cos1) * (z**2 - unit1)

    return R
