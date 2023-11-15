import numpy as np
from scipy.special import lpn
from scipy.special import lpmn
import treecorr
from classy import Class
from scipy import interpolate

def xi_theta_bin_averaged(theta_min, theta_max, ell):
    '''
    theta_min and theta_max are in radians
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (65) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below

    x_min = np.cos(theta_min)
    x_max = np.cos(theta_max)

    P_ell_x_min_array = lpn(ell[-1]+1,x_min)[0] # all the (first kind) Legendre polynomials of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_min_array = P_ell_x_min_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_lm1_x_min_array = P_ell_x_min_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below

    P_ell_x_max_array = lpn(ell[-1]+1,x_max)[0]
    P_lp1_x_max_array = P_ell_x_max_array[3:]
    P_lm1_x_max_array = P_ell_x_max_array[1:-2]

    P_l_bin_averaged = 1./(2.*l+1.)/(x_min - x_max)*(P_lp1_x_min_array-P_lm1_x_min_array-(P_lp1_x_max_array-P_lm1_x_max_array))

    return P_l_bin_averaged
    #return np.sum( (2.*l+1) / (4*np.pi) * P_l_bin_averaged * C_l )

def xit_theta_bin_averaged(theta_min, theta_max, ell):
    '''
    theta_min and theta_max is in radians
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (B2) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below

    x_min = np.cos(theta_min)
    x_max = np.cos(theta_max)

    P_ell_x_min_array = lpn(ell[-1]+1,x_min)[0] # all the (first kind) Legendre polynomials of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_min_array = P_ell_x_min_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_min_array = P_ell_x_min_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_min_array = P_ell_x_min_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below

    P_ell_x_max_array = lpn(ell[-1]+1,x_max)[0] # all the (first kind) Legendre polynomials of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_max_array = P_ell_x_max_array[3:] 
    P_l_x_max_array = P_ell_x_max_array[2:-1] 
    P_lm1_x_max_array = P_ell_x_max_array[1:-2] 

    P_l_2_bin_averaged = 1./(x_min - x_max)*(
                        (l+2./(2*l+1))*(P_lm1_x_min_array-P_lm1_x_max_array)
                        +(2.-l)*(x_min*P_l_x_min_array-x_max*P_l_x_max_array)
                        -(2./(2*l+1))*(P_lp1_x_min_array-P_lp1_x_max_array)
                        )

    return P_l_2_bin_averaged
    #return np.sum( (2.*l+1) / (4*np.pi) * P_l_2_bin_averaged / (l*(l+1)) * C_l )

def xip_theta_bin_averaged(theta_min, theta_max, ell):
    '''
    theta_min and theta_max is in radians    
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (B5) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    here, G_l_2_x_p = G_l_2_+(x) + G_l_2_-(x) i.e. LHS of Friedrich++ for the 'p'lus middle sign
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below

    x_min = np.cos(theta_min)
    x_max = np.cos(theta_max)

    P_ell_x_min_array, ddx_P_ell_x_min_array = lpn(ell[-1]+1,x_min) # all the (first kind) Legendre polynomials and derivatives of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_min_array = P_ell_x_min_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_min_array = P_ell_x_min_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_min_array = P_ell_x_min_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below
    ddx_P_l_x_min_array = ddx_P_ell_x_min_array[2:-1] 
    ddx_P_lm1_x_min_array = ddx_P_ell_x_min_array[1:-2] 

    P_ell_x_max_array, ddx_P_ell_x_max_array = lpn(ell[-1]+1,x_max) # all the (first kind) Legendre polynomials and derivatives of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_max_array = P_ell_x_max_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_max_array = P_ell_x_max_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_max_array = P_ell_x_max_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below
    ddx_P_l_x_max_array = ddx_P_ell_x_max_array[2:-1] 
    ddx_P_lm1_x_max_array = ddx_P_ell_x_max_array[1:-2] 

    G_l_2_x_p_bin_averaged = 1./(x_min - x_max)*(
                            -l*(l-1.)/2.*(l+2./(2.*l+1.))*(P_lm1_x_min_array-P_lm1_x_max_array)
                            -l*(l-1.)*(2.-l)/2.*(x_min*P_l_x_min_array-x_max*P_l_x_max_array)
                            +l*(l-1.)/(2.*l+1.)*(P_lp1_x_min_array-P_lp1_x_max_array)
                            +(4.-l)*(ddx_P_l_x_min_array-ddx_P_l_x_max_array)
                            +(l+2.)*(x_min*ddx_P_lm1_x_min_array-x_max*ddx_P_lm1_x_max_array-(P_lm1_x_min_array-P_lm1_x_max_array))
                            +2.*(l-1.)*(x_min*ddx_P_l_x_min_array-x_max*ddx_P_l_x_max_array-(P_l_x_min_array-P_l_x_max_array))
                            -2.*(l+2.)*(ddx_P_lm1_x_min_array-ddx_P_lm1_x_max_array)
                             )

    return G_l_2_x_p_bin_averaged
    #return np.sum( (2.*l+1) / (4*np.pi) * 2. * G_l_2_x_p_bin_averaged / (l*l*(l+1.)*(l+1.)) * C_l )

def xim_theta_bin_averaged(theta_min, theta_max, ell):
    '''
    theta_min and theta_max is in radians    
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (B5) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    here, G_l_2_x_m = G_l_2_+(x) - G_l_2_-(x) i.e. LHS of Friedrich++ for the 'm'inus middle sign
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below

    x_min = np.cos(theta_min)
    x_max = np.cos(theta_max)

    P_ell_x_min_array, ddx_P_ell_x_min_array = lpn(ell[-1]+1,x_min) # all the (first kind) Legendre polynomials and derivatives of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_min_array = P_ell_x_min_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_min_array = P_ell_x_min_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_min_array = P_ell_x_min_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below
    ddx_P_l_x_min_array = ddx_P_ell_x_min_array[2:-1] 
    ddx_P_lm1_x_min_array = ddx_P_ell_x_min_array[1:-2] 

    P_ell_x_max_array, ddx_P_ell_x_max_array = lpn(ell[-1]+1,x_max) # all the (first kind) Legendre polynomials and derivatives of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_max_array = P_ell_x_max_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_max_array = P_ell_x_max_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_max_array = P_ell_x_max_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below
    ddx_P_l_x_max_array = ddx_P_ell_x_max_array[2:-1] 
    ddx_P_lm1_x_max_array = ddx_P_ell_x_max_array[1:-2] 

    G_l_2_x_m_bin_averaged = 1./(x_min - x_max)*(
                            -l*(l-1.)/2.*(l+2./(2.*l+1.))*(P_lm1_x_min_array-P_lm1_x_max_array)
                            -l*(l-1.)*(2.-l)/2.*(x_min*P_l_x_min_array-x_max*P_l_x_max_array)
                            +l*(l-1.)/(2.*l+1.)*(P_lp1_x_min_array-P_lp1_x_max_array)
                            +(4.-l)*(ddx_P_l_x_min_array-ddx_P_l_x_max_array)
                            +(l+2.)*(x_min*ddx_P_lm1_x_min_array-x_max*ddx_P_lm1_x_max_array-(P_lm1_x_min_array-P_lm1_x_max_array))
                            -2.*(l-1.)*(x_min*ddx_P_l_x_min_array-x_max*ddx_P_l_x_max_array-(P_l_x_min_array-P_l_x_max_array))
                            +2.*(l+2.)*(ddx_P_lm1_x_min_array-ddx_P_lm1_x_max_array)
                             )

    return G_l_2_x_m_bin_averaged
    #return np.sum( (2.*l+1) / (4*np.pi) * 2. * G_l_2_x_m_bin_averaged / (l*l*(l+1.)*(l+1.)) * C_l )

###################################

ell = np.arange(0, 15000-1)

min_sep_tc_xi = 5
max_sep_tc_xi = 250
nbins_tc_xi = 15

kk_xi = treecorr.KKCorrelation(min_sep=min_sep_tc_xi, max_sep=max_sep_tc_xi, nbins=nbins_tc_xi, sep_units='arcmin')
alpha_arcmins_xi = kk_xi.rnom
alpha_min_arcmins_xi = kk_xi.left_edges
alpha_max_arcmins_xi = kk_xi.right_edges

xip_bin_averaged_values = np.zeros([alpha_arcmins_xi.size, ell.size-2])
xim_bin_averaged_values = np.zeros([alpha_arcmins_xi.size, ell.size-2])
xit_bin_averaged_values = np.zeros([alpha_arcmins_xi.size, ell.size-2])
xi_bin_averaged_values = np.zeros([alpha_arcmins_xi.size, ell.size-2])

for i in range(alpha_arcmins_xi.size):
    xip_bin_averaged_values[i,:] = xip_theta_bin_averaged(np.radians(alpha_min_arcmins_xi[i]/60), np.radians(alpha_max_arcmins_xi[i]/60), ell)
    xim_bin_averaged_values[i,:] = xim_theta_bin_averaged(np.radians(alpha_min_arcmins_xi[i]/60), np.radians(alpha_max_arcmins_xi[i]/60), ell)
    xit_bin_averaged_values[i,:] = xit_theta_bin_averaged(np.radians(alpha_min_arcmins_xi[i]/60), np.radians(alpha_max_arcmins_xi[i]/60), ell)
    xi_bin_averaged_values[i,:] = xi_theta_bin_averaged(np.radians(alpha_min_arcmins_xi[i]/60), np.radians(alpha_max_arcmins_xi[i]/60), ell)

np.save('./data/angular_bins/xip_'+str(min_sep_tc_xi)+'_'+str(max_sep_tc_xi)+'_'+str(nbins_tc_xi)+'_bin_averaged_values', xip_bin_averaged_values)
np.save('./data/angular_bins/xim_'+str(min_sep_tc_xi)+'_'+str(max_sep_tc_xi)+'_'+str(nbins_tc_xi)+'_bin_averaged_values', xim_bin_averaged_values)
np.save('./data/angular_bins/xit_'+str(min_sep_tc_xi)+'_'+str(max_sep_tc_xi)+'_'+str(nbins_tc_xi)+'_bin_averaged_values', xit_bin_averaged_values)
np.save('./data/angular_bins/xi_'+str(min_sep_tc_xi)+'_'+str(max_sep_tc_xi)+'_'+str(nbins_tc_xi)+'_bin_averaged_values', xi_bin_averaged_values)

W = 90

min_sep_tc = 5
max_sep_tc = 2*W-10
nbins_tc = 15

kk = treecorr.KKCorrelation(min_sep=min_sep_tc, max_sep=max_sep_tc, nbins=nbins_tc, sep_units='arcmin')
alpha_arcmins = kk.rnom
alpha_min_arcmins = kk.left_edges
alpha_max_arcmins = kk.right_edges

zetap_bin_averaged_values = np.zeros([alpha_arcmins.size, ell.size-2])
zetam_bin_averaged_values = np.zeros([alpha_arcmins.size, ell.size-2])
zetat_bin_averaged_values = np.zeros([alpha_arcmins.size, ell.size-2])
zeta_bin_averaged_values = np.zeros([alpha_arcmins.size, ell.size-2])

for i in range(alpha_arcmins.size):
    zetap_bin_averaged_values[i,:] = xip_theta_bin_averaged(np.radians(alpha_min_arcmins[i]/60), np.radians(alpha_max_arcmins[i]/60), ell)
    zetam_bin_averaged_values[i,:] = xim_theta_bin_averaged(np.radians(alpha_min_arcmins[i]/60), np.radians(alpha_max_arcmins[i]/60), ell)
    zetat_bin_averaged_values[i,:] = xit_theta_bin_averaged(np.radians(alpha_min_arcmins[i]/60), np.radians(alpha_max_arcmins[i]/60), ell)
    zeta_bin_averaged_values[i,:] = xi_theta_bin_averaged(np.radians(alpha_min_arcmins[i]/60), np.radians(alpha_max_arcmins[i]/60), ell)

np.save('./data/angular_bins/zetap_W'+str(W)+'_'+str(min_sep_tc)+'_'+str(max_sep_tc)+'_'+str(nbins_tc)+'_bin_averaged_values', zetap_bin_averaged_values)
np.save('./data/angular_bins/zetam_W'+str(W)+'_'+str(min_sep_tc)+'_'+str(max_sep_tc)+'_'+str(nbins_tc)+'_bin_averaged_values', zetam_bin_averaged_values)
np.save('./data/angular_bins/zetat_W'+str(W)+'_'+str(min_sep_tc)+'_'+str(max_sep_tc)+'_'+str(nbins_tc)+'_bin_averaged_values', zetat_bin_averaged_values)
np.save('./data/angular_bins/zeta_W'+str(W)+'_'+str(min_sep_tc)+'_'+str(max_sep_tc)+'_'+str(nbins_tc)+'_bin_averaged_values', zeta_bin_averaged_values)

'''
##########################################################
# create helper functions from Class object

Omega_b = 0.046
Omega_cdm = 0.233
h = 0.7
sigma8 = 0.82
#A_s = 2.19685e-9
A_s = np.exp(np.log(10**10 * 2.19685e-9))/(10**10)
n_s = 0.97
w_0 = -1.0
w_a = 0.0
c_min = 3.13 # emu_dmonly (fiducial)
eta_0 = 0.603 # emu_dmonly (fiducial)

omega_b = Omega_b*h*h
omega_cdm = Omega_cdm*h*h

Omega_m = Omega_cdm + Omega_b

commonsettings_nl  = {
            'h':h,
            'omega_b':omega_b,
            'omega_cdm': omega_cdm,
            'A_s':A_s,
            'n_s':n_s,
            'Omega_Lambda':0.0,
            'fluid_equation_of_state':'CLP',
            'w0_fld':w_0,
            'wa_fld':w_a,
            'output':'mPk',
            #'P_k_max_1/Mpc':500.0,
            #'z_max_pk':3.0,
            #'non linear':'halofit',
            'non linear':'hmcode',
            'eta_0':eta_0,
            'c_min':c_min,
            }

ClassyClassObject_nl = Class()
ClassyClassObject_nl.set(commonsettings_nl)
ClassyClassObject_nl.compute()

chi_z = interpolate.interp1d(ClassyClassObject_nl.get_background()['z'], ClassyClassObject_nl.get_background()['comov. dist.'], kind='cubic', fill_value=0.0)
z_chi = interpolate.interp1d(ClassyClassObject_nl.get_background()['comov. dist.'], ClassyClassObject_nl.get_background()['z'], kind='cubic', fill_value=0.0)
H_z = interpolate.interp1d(ClassyClassObject_nl.get_background()['z'], ClassyClassObject_nl.get_background()['H [1/Mpc]'], kind='cubic', fill_value=0.0)
D_plus_z = interpolate.interp1d(ClassyClassObject_nl.get_background()['z'], ClassyClassObject_nl.get_background()['gr.fac. D'], kind='cubic', fill_value=0.0) # normalised to 1 at z=0

# for LOS projection kernels
H_0 = h*100
Omega0_m = Omega_m

z_min = 0.01
z_max = 2.0
z_bins_los = 200

z_array_los = np.linspace(z_min, z_max, z_bins_los)

chi_inv_z_array_los = 1./chi_z(z_array_los)
H_inv_z_array_los = 1./H_z(z_array_los)
D_plus_z_array_los = D_plus_z(z_array_los)

np.save('./data/chi_inv_z_array_los_'+str(z_min)+'_'+str(z_max)+'_'+str(z_bins_los), chi_inv_z_array_los)
np.save('./data/H_inv_z_array_los_'+str(z_min)+'_'+str(z_max)+'_'+str(z_bins_los), H_inv_z_array_los)
np.save('./data/D_plus_z_array_los_'+str(z_min)+'_'+str(z_max)+'_'+str(z_bins_los), D_plus_z_array_los)
'''

