import numpy as np
from perturbation_kernels import F2_EdS, S2

def is_triangle_closed(k_1, k_2, k_3):
    # test whether the input k values form a closed triangle or not (necessary condition for bispectrum)
    k = np.array([k_1, k_2, k_3]).T
    k.sort(axis=1)

    mask = np.array(np.ones(k_1.size), dtype=bool) # array of True values  
    mask[k[:,2] > k[:,0] + k[:,1]] = False

    return mask

######################################################################################

## tree-level SPT matter bispectrum

def B_tree(k_1, k_2, k_3, Pk_1, Pk_2, Pk_3):
    # Evaluate the tree-level bispctrum at a given redshift --> quantities like Pk_1 etc. should already have been computed at the given redshift

    # Already ensure beforehand that the input k_i modes form a closed triangle
    # k [1/Mpc] and P(k,z) [Mpc^3]

    B = 2.0*(F2_EdS(k_1,k_2,k_3)*Pk_1*Pk_2 + F2_EdS(k_2,k_3,k_1)*Pk_2*Pk_3 + F2_EdS(k_3,k_1,k_2)*Pk_3*Pk_1) 

    return B

######################################################################################

## response function squeezed bispectrum

def B_RF(k_s, k_m, k_h, Pk_s, Pk_h, n_h, G_1_h, G_K_h):
    # Evaluate the squeezed Response funtion bispctrum at a given redshift --> quantities like Pk_h etc. should already have been computed at the given redshift

    # Already ensure beforehand that the input k_i modes form a closed triangle AND are already sorted i.e. k_s < k_m < k_h

    # - negative sign in front to get cosine of the angle between the hard and soft vectors and not the triangular sides
    #mu_h_s = - 0.5*(k_m*k_m + k_s*k_s - k_h*k_h) / (k_m*k_s); # cosine of the angle between k_m and k_s
    mu_h_s = - 0.5*(k_h*k_h + k_s*k_s - k_m*k_m) / (k_h*k_s); # cosine of the angle between k_h and k_s - is this more correct?

    R_1_h = 1.0 - n_h/3.0 + G_1_h
    R_K_h = G_K_h - n_h

    B = (R_1_h + (mu_h_s*mu_h_s - 1.0/3.0)*R_K_h )*Pk_h*Pk_s

    return B

######################################################################################

## matter bispectrum fitting formulae

def Q3(n):
    val = ( 4. - 2.**n ) / (1. + 2.**(n+1) )

    if (val < 0):
        print( "Q3 is negative with value = "+str(val)+" for n = "+ str(n) +"\n")
        return 0.0
    else:
        return val

def F2_eff(k_1, k_2, k_3, a_1, a_2, b_1, b_2, c_1, c_2):
    cos_phi_12 = 0.5*(k_3*k_3-k_1*k_1-k_2*k_2)/(k_1*k_2) # law of cosines (with a negative sign to get the complementary angle outside the triangle)
    return 5./7.*a_1*a_2 + 0.5*cos_phi_12*(k_1/k_2 + k_2/k_1)*b_1*b_2 + 2./7.*cos_phi_12*cos_phi_12*c_1*c_2

## Gil-Marin matter bispectrum fitting formula

a1_GM =  0.484
a2_GM =  3.740
a3_GM = -0.849
a4_GM =  0.392
a5_GM =  1.013
a6_GM = -0.575
a7_GM =  0.128
a8_GM = -0.722
a9_GM = -0.926

def a_GM(n, q, sigma8_z):
    # n(k) is the logarithmic tilt of the P_lin(k) 
    # q(z) = k/k_nl(z) 
    # sigma8(z) is the sigma8_0 extrapolated to redshift z with D(z)
    return ( 1. + sigma8_z**a6_GM*np.sqrt(0.7*Q3(n))*(q*a1_GM)**(n+a2_GM) ) / ( 1. + (q*a1_GM)**(n+a2_GM) )

def b_GM(n, q):
    # n(k) is the logarithmic tilt of the P_lin(k) 
    # q(z) = k/k_nl(z) 
    return ( 1. + 0.2*a3_GM*(n+3)*(q*a7_GM)**(n+3+a8_GM) ) / ( 1. + (q*a7_GM)**(n+3.5+a8_GM) )

def c_GM(n, q):
    # n(k) is the logarithmic tilt of the P_lin(k) 
    # q(z) = k/k_nl(z) 
    return ( 1. + 4.5*a4_GM/(1.5 + (n+3)**4) * (q*a5_GM)**(n+3+a9_GM) ) / ( 1. + (q*a5_GM)**(n+3.5+a9_GM) )

def B_GM(k_1, k_2, k_3, Pk_1, Pk_2, Pk_3, a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3):
    # Evaluate the GM bispctrum at a given redshift --> quantities like Pk_1 etc. should already have been computed at the given redshift

    # Already ensure beforehand that the input k_i modes form a closed triangle
    # k [1/Mpc] and P(k,z) [Mpc^3]

    B = 2.0*(F2_eff(k_1,k_2,k_3,a_1,a_2,b_1,b_2,c_1,c_2)*Pk_1*Pk_2
            +F2_eff(k_2,k_3,k_1,a_2,a_3,b_2,b_3,c_2,c_3)*Pk_2*Pk_3
            +F2_eff(k_3,k_1,k_2,a_3,a_1,b_3,b_1,c_3,c_1)*Pk_3*Pk_1)

    return B

######################################################################################

# components for tracer bispectrum (without the bias coefficients)

def B_PaPb(Pk_a, Pk_b):
    # Evaluate the P(k_a)P(k_b) component of the tracer bispctrum at a given redshift --> 
    # quantities like Pk_a etc. should already have been computed at the given redshift

    return Pk_a*Pk_b

def B_S2PaPb(k_a, k_b, k_c, Pk_a, Pk_b):
    # Evaluate the S2(k_a,k_b)P(k_a)P(k_b) component of the tracer bispctrum at a given redshift --> 
    # quantities like Pk_a etc. should already have been computed at the given redshift

    return S2(k_a,k_b,k_c)*Pk_a*Pk_b

def B_PP(Pk_a, Pk_b, Pk_c):
    # Evaluate the P(k_a)P(k_b) + P(k_a)P(k_c) + P(k_b)P(k_c) component of the tracer bispctrum at a given redshift --> 
    # quantities like Pk_a etc. should already have been computed at the given redshift

    return Pk_a*Pk_b + Pk_a*Pk_c + Pk_b*Pk_c

def B_S2PP(k_a, k_b, k_c, Pk_a, Pk_b, Pk_c):
    # Evaluate the S2(k_a,k_b)P(k_a)P(k_b) + S2(k_a,k_c)P(k_a)P(k_c) + S2(k_b,k_c)P(k_b)P(k_c) component of the tracer bispctrum at a given redshift --> 
    # quantities like Pk_a etc. should already have been computed at the given redshift

    return S2(k_a,k_b,k_c)*Pk_a*Pk_b + S2(k_a,k_c,k_b)*Pk_a*Pk_c + S2(k_b,k_c,k_a)*Pk_b*Pk_c