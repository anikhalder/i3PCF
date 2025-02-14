### example call to execute script
# python main.py 0 1

import time
start_program = time.time()

import numpy as np
import os
import platform
import itertools
import multiprocessing as mp
from neutrino_utils import *
from classy import Class
from cosmology_utils import CosmoClass
from integratedBispectrum2D_grid import *
import treecorr
from projection_kernels import *
from scipy import interpolate
from real_space_2D import *
import constants
from misc_utils import num_correlations, eta_0_val
import input
import sys
from window_utils import iZ_A2pt, iZ_A2pt_bin_averaged

try:
    import healpy as hp
    healpy_installed = True
except:
    print('!!! Encountered error while importing healpy !!!', flush=True)
    healpy_installed = False
 
#################################################################################################################################
#################################################################################################################################
# STEP 0: Overall settings for performing computations, saving calculations etc.
#################################################################################################################################
#################################################################################################################################

### arguments for the script call
start_idx = int(sys.argv[1])
stop_idx = int(sys.argv[2])

nside = input.nside # If not None, pixel window function will be considered for C_ells

P3D_type = input.P3D_type
compute_P_grid = input.compute_P_grid
compute_P_spectra_and_correlations = input.compute_P_spectra_and_correlations
B3D_type = input.B3D_type
compute_B_grid = input.compute_B_grid
compute_B_spectra = input.compute_B_spectra
compute_iB_grid = input.compute_iB_grid
compute_iB_spectra_and_correlations = input.compute_iB_spectra_and_correlations
compute_A2pt = input.compute_A2pt
compute_A2pt_bin_averaged = input.compute_A2pt_bin_averaged
compute_H_chi_D = input.compute_H_chi_D

halo_type_user = input.halo_type_user

xi_correlation_name_list = input.xi_correlation_name_list
iZ_correlation_name_list = input.iZ_correlation_name_list

#####################
# (l,z) grid settings
l_array = input.l_array
z_array = input.z_array

l_B_array = input.l_B_array
z_B_array = input.z_B_array

#### Window functions ####

# radius of the compensated and tophat filters
theta_U_arcmins = input.theta_U_arcmins
theta_T_arcmins = input.theta_T_arcmins
theta_U = np.radians(theta_U_arcmins/60.0) # this is the scale of the aperture for computing mean mass within a patch (e.g. aperture mass, mean galaxy density)
theta_T = np.radians(theta_T_arcmins/60.0) # this is the size of the tophat aperture for computing position-dependent 2PCF

#####################

if (compute_P_spectra_and_correlations == 'yes' or compute_B_spectra == 'yes' or compute_iB_spectra_and_correlations == 'yes'):

    spectra_and_correlation_type = input.spectra_and_correlation_type

    if ('halo' in spectra_and_correlation_type or 'galaxy' in spectra_and_correlation_type):
    
        LENS_BIN_NAME = input.LENS_BIN_NAME
        halo_type = input.halo_type_user

    if ('kappa' in spectra_and_correlation_type or 'shear' in spectra_and_correlation_type):

        SOURCE_BIN_NAME_LIST = input.SOURCE_BIN_NAME_LIST
        SOURCE_BIN_VALUES = input.SOURCE_BIN_VALUES
        
        SOURCE_BIN_delta_photoz_values = input.SOURCE_BIN_delta_photoz_values
        SOURCE_BIN_m_values = input.SOURCE_BIN_m_values

        num_2pt_sss_correlations = num_correlations(len(SOURCE_BIN_NAME_LIST), 2)
        num_i3pt_sss_correlations = num_correlations(len(SOURCE_BIN_NAME_LIST), 3)

        print('Number of 2pt shear x shear correlations (for plus or minus or kappa) =', num_2pt_sss_correlations, flush=True)
        print('Number of i3pt shear x shear correlations (for plus or minus or kappa) =', num_i3pt_sss_correlations, flush=True)

# compute and load information relevant for angular bin operations
if (compute_P_spectra_and_correlations == 'yes' or compute_iB_spectra_and_correlations == 'yes'):
    
    spectra_and_correlation_type = input.spectra_and_correlation_type

    angular_bins_path = input.angular_bins_path

    ## set the angular bins in which to compute the global 2PCFs

    min_sep_tc_xi = input.min_sep_tc_xi
    max_sep_tc_xi = input.max_sep_tc_xi
    nbins_tc_xi = input.nbins_tc_xi

    kk_xi = treecorr.KKCorrelation(min_sep=min_sep_tc_xi, max_sep=max_sep_tc_xi, nbins=nbins_tc_xi, sep_units='arcmin')
    alpha_arcmins_xi = kk_xi.rnom
    alpha_min_arcmins_xi = kk_xi.left_edges
    alpha_max_arcmins_xi = kk_xi.right_edges
    
    np.savetxt(angular_bins_path+'alpha_xi_angles_cen_arcmins_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bins.tab', alpha_arcmins_xi.T)
    np.savetxt(angular_bins_path+'alpha_xi_angles_min_arcmins_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bins.tab', alpha_min_arcmins_xi.T)
    np.savetxt(angular_bins_path+'alpha_xi_angles_max_arcmins_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bins.tab', alpha_max_arcmins_xi.T)

    ## set the angular bins in which to compute the local 2PCFs
    min_sep_tc = input.min_sep_tc
    max_sep_tc = input.max_sep_tc
    nbins_tc = input.nbins_tc

    kk = treecorr.KKCorrelation(min_sep=min_sep_tc, max_sep=max_sep_tc, nbins=nbins_tc, sep_units='arcmin')
    alpha_arcmins = kk.rnom
    alpha_min_arcmins = kk.left_edges
    alpha_max_arcmins = kk.right_edges

    np.savetxt(angular_bins_path+'alpha_iZ_W'+str(theta_T_arcmins)+'_angles_cen_arcmins_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bins.tab', alpha_arcmins.T)
    np.savetxt(angular_bins_path+'alpha_iZ_W'+str(theta_T_arcmins)+'_angles_min_arcmins_'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bins.tab', alpha_min_arcmins.T)
    np.savetxt(angular_bins_path+'alpha_iZ_W'+str(theta_T_arcmins)+'_angles_max_arcmins_'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bins.tab', alpha_max_arcmins.T)

    ## pre-compute or load the bin-averaged values

    if (os.path.exists(angular_bins_path+'xip_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bin_averaged_values.npy') == False or
        os.path.exists(angular_bins_path+'iZp_W'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bin_averaged_values.npy') == False):

        print('Computing bin-averaged values of Legendre polynomials for xi and iZ', flush=True)

        ell = np.arange(0, constants._l_max_-1)

        xip_bin_averaged_values = np.zeros([alpha_arcmins_xi.size, ell.size-2])
        xim_bin_averaged_values = np.zeros([alpha_arcmins_xi.size, ell.size-2])
        xit_bin_averaged_values = np.zeros([alpha_arcmins_xi.size, ell.size-2])
        xi_bin_averaged_values = np.zeros([alpha_arcmins_xi.size, ell.size-2])

        for i in range(alpha_arcmins_xi.size):
            xip_bin_averaged_values[i,:] = xip_theta_bin_averaged_Legendre(np.radians(alpha_min_arcmins_xi[i]/60), np.radians(alpha_max_arcmins_xi[i]/60), ell)
            xim_bin_averaged_values[i,:] = xim_theta_bin_averaged_Legendre(np.radians(alpha_min_arcmins_xi[i]/60), np.radians(alpha_max_arcmins_xi[i]/60), ell)
            xit_bin_averaged_values[i,:] = xit_theta_bin_averaged_Legendre(np.radians(alpha_min_arcmins_xi[i]/60), np.radians(alpha_max_arcmins_xi[i]/60), ell)
            xi_bin_averaged_values[i,:] = xi_theta_bin_averaged_Legendre(np.radians(alpha_min_arcmins_xi[i]/60), np.radians(alpha_max_arcmins_xi[i]/60), ell)

        np.save(angular_bins_path+'xip_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bin_averaged_values', xip_bin_averaged_values)
        np.save(angular_bins_path+'xim_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bin_averaged_values', xim_bin_averaged_values)
        np.save(angular_bins_path+'xit_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bin_averaged_values', xit_bin_averaged_values)
        np.save(angular_bins_path+'xi_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bin_averaged_values', xi_bin_averaged_values)

        iZp_bin_averaged_values = np.zeros([alpha_arcmins.size, ell.size-2])
        iZm_bin_averaged_values = np.zeros([alpha_arcmins.size, ell.size-2])
        iZt_bin_averaged_values = np.zeros([alpha_arcmins.size, ell.size-2])
        iZ_bin_averaged_values = np.zeros([alpha_arcmins.size, ell.size-2])

        for i in range(alpha_arcmins.size):
            iZp_bin_averaged_values[i,:] = xip_theta_bin_averaged_Legendre(np.radians(alpha_min_arcmins[i]/60), np.radians(alpha_max_arcmins[i]/60), ell)
            iZm_bin_averaged_values[i,:] = xim_theta_bin_averaged_Legendre(np.radians(alpha_min_arcmins[i]/60), np.radians(alpha_max_arcmins[i]/60), ell)
            iZt_bin_averaged_values[i,:] = xit_theta_bin_averaged_Legendre(np.radians(alpha_min_arcmins[i]/60), np.radians(alpha_max_arcmins[i]/60), ell)
            iZ_bin_averaged_values[i,:] = xi_theta_bin_averaged_Legendre(np.radians(alpha_min_arcmins[i]/60), np.radians(alpha_max_arcmins[i]/60), ell)

        np.save(angular_bins_path+'iZp_W'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bin_averaged_values', iZp_bin_averaged_values)
        np.save(angular_bins_path+'iZm_W'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bin_averaged_values', iZm_bin_averaged_values)
        np.save(angular_bins_path+'iZt_W'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bin_averaged_values', iZt_bin_averaged_values)
        np.save(angular_bins_path+'iZ_W'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bin_averaged_values', iZ_bin_averaged_values)

    print('Loading pe-computed bin-averaged values of Legendre polynomials for xi and iZ', flush=True)

    xip_theta_bin_averaged_values = np.load(angular_bins_path+'xip_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bin_averaged_values.npy')
    xim_theta_bin_averaged_values = np.load(angular_bins_path+'xim_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bin_averaged_values.npy')
    xit_theta_bin_averaged_values = np.load(angular_bins_path+'xit_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bin_averaged_values.npy')
    xi_theta_bin_averaged_values = np.load(angular_bins_path+'xi_'+str(int(min_sep_tc_xi))+'_'+str(int(max_sep_tc_xi))+'_'+str(nbins_tc_xi)+'_bin_averaged_values.npy')

    iZp_theta_bin_averaged_values = np.load(angular_bins_path+'iZp_W'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bin_averaged_values.npy')
    iZm_theta_bin_averaged_values = np.load(angular_bins_path+'iZm_W'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bin_averaged_values.npy')
    iZt_theta_bin_averaged_values = np.load(angular_bins_path+'iZt_W'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bin_averaged_values.npy')
    iZ_theta_bin_averaged_values = np.load(angular_bins_path+'iZ_W'+str(theta_T_arcmins)+'_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'_bin_averaged_values.npy')

    def xip_theta_bin_averaged(ell, C_ell):
        l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
        C_l = C_ell[2:]
        G_l_2_x_p_bin_averaged = xip_theta_bin_averaged_values
        return np.sum( ((2.*l+1) / (4*np.pi) * 2. * G_l_2_x_p_bin_averaged / (l*l*(l+1.)*(l+1.)) * C_l), axis=1 )

    def xim_theta_bin_averaged(ell, C_ell):
        l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
        C_l = C_ell[2:]
        G_l_2_x_m_bin_averaged = xim_theta_bin_averaged_values
        return np.sum( ((2.*l+1) / (4*np.pi) * 2. * G_l_2_x_m_bin_averaged / (l*l*(l+1.)*(l+1.)) * C_l), axis=1 )

    def xit_theta_bin_averaged(ell, C_ell):
        l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
        C_l = C_ell[2:]
        P_l_2_bin_averaged = xit_theta_bin_averaged_values
        return np.sum( ((2.*l+1) / (4*np.pi) * P_l_2_bin_averaged / (l*(l+1)) * C_l), axis=1 )

    def xi_theta_bin_averaged(ell, C_ell):
        l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
        C_l = C_ell[2:]
        P_l_bin_averaged = xi_theta_bin_averaged_values
        return np.sum( ((2.*l+1) / (4*np.pi) * P_l_bin_averaged * C_l), axis=1 )

    def iZp_theta_bin_averaged(ell, iB_ell):
        l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
        iB_l = iB_ell[2:]
        G_l_2_x_p_bin_averaged = iZp_theta_bin_averaged_values
        return np.sum( ((2.*l+1) / (4*np.pi) * 2. * G_l_2_x_p_bin_averaged / (l*l*(l+1.)*(l+1.)) * iB_l), axis=1 )

    def iZm_theta_bin_averaged(ell, iB_ell):
        l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
        iB_l = iB_ell[2:]
        G_l_2_x_m_bin_averaged = iZm_theta_bin_averaged_values
        return np.sum( ((2.*l+1) / (4*np.pi) * 2. * G_l_2_x_m_bin_averaged / (l*l*(l+1.)*(l+1.)) * iB_l), axis=1 )

    def iZt_theta_bin_averaged(ell, iB_ell):
        l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
        iB_l = iB_ell[2:]
        P_l_2_bin_averaged = iZt_theta_bin_averaged_values
        return np.sum( ((2.*l+1) / (4*np.pi) * P_l_2_bin_averaged / (l*(l+1)) * iB_l), axis=1 )

    def iZ_theta_bin_averaged(ell, iB_ell):
        l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
        iB_l = iB_ell[2:]
        P_l_bin_averaged = iZ_theta_bin_averaged_values
        return np.sum( ((2.*l+1) / (4*np.pi) * P_l_bin_averaged * iB_l), axis=1 )
    
cosmo_parameters_output_path = input.cosmo_parameters_output_path
if (os.path.isdir(cosmo_parameters_output_path) == False):
    os.mkdir(cosmo_parameters_output_path)
    
P_l_z_grid_path = input.P_l_z_grid_path
iB_l_z_grid_path = input.iB_l_z_grid_path
B_l1_l2_l3_z_grid_path = input.B_l1_l2_l3_z_grid_path

if (compute_P_grid == 'yes'):
    if (os.path.isdir(P_l_z_grid_path) == False):
        os.mkdir(P_l_z_grid_path)

    np.savetxt(P_l_z_grid_path+'l_array.tab', l_array.T)
    np.savetxt(P_l_z_grid_path+'z_array.tab', z_array.T)

if (compute_B_grid == 'yes'):
    if (os.path.isdir(B_l1_l2_l3_z_grid_path) == False):
        os.mkdir(B_l1_l2_l3_z_grid_path)

    np.savetxt(B_l1_l2_l3_z_grid_path+'l_B_array.tab', l_B_array.T)
    np.savetxt(B_l1_l2_l3_z_grid_path+'z_B_array.tab', z_B_array.T)

if (compute_iB_grid == 'yes'):
    if (os.path.isdir(iB_l_z_grid_path) == False):
        os.mkdir(iB_l_z_grid_path)
    
    np.savetxt(iB_l_z_grid_path+'l_array.tab', l_array.T)
    np.savetxt(iB_l_z_grid_path+'z_array.tab', z_array.T)

if (compute_P_spectra_and_correlations == 'yes'):

    P_spectra_path = input.P_spectra_path

    if (os.path.isdir(P_spectra_path) == False):
        os.mkdir(P_spectra_path)

    xi_correlations_path = input.xi_correlations_path

    if (os.path.isdir(xi_correlations_path) == False):
        os.mkdir(xi_correlations_path)

if (compute_B_spectra == 'yes'):

    B_spectra_path = input.B_spectra_path

    if (os.path.isdir(B_spectra_path) == False):
        os.mkdir(B_spectra_path)

if (compute_iB_spectra_and_correlations == 'yes'):

    iB_spectra_path = input.iB_spectra_path
    if (os.path.isdir(iB_spectra_path) == False):
        os.mkdir(iB_spectra_path)

    iZ_correlations_path = input.iZ_correlations_path
    if (os.path.isdir(iZ_correlations_path) == False):
        os.mkdir(iZ_correlations_path)

if (compute_A2pt == 'yes' or compute_A2pt_bin_averaged == 'yes'):
    print('Computing the area pre-factors for iZ', flush=True)

    A2pt_path = input.A2pt_path
    if (os.path.isdir(A2pt_path) == False):
        os.mkdir(A2pt_path)

    ## set the angular bins in which to compute the local 2PCFs
    area_compute_time = time.time()
    
    min_sep_tc = input.min_sep_tc
    max_sep_tc = input.max_sep_tc
    nbins_tc = input.nbins_tc
    
    binedges = np.radians(np.geomspace(min_sep_tc,max_sep_tc,nbins_tc+1)/60)
    bincenters = np.sqrt(binedges[1:]*binedges[:-1])

    A2pt = np.zeros([nbins_tc])
    for i, alpha in enumerate(bincenters):
        A2pt[i] = iZ_A2pt(alpha, theta_T)
    
    if (compute_A2pt_bin_averaged == 'yes'): 
        A2pt_bin_averaged = iZ_A2pt_bin_averaged(binedges, theta_T)
        dat = np.array([bincenters, binedges[:-1], binedges[1:], A2pt, A2pt_bin_averaged])
        np.savetxt(A2pt_path+'A2pt_bin_averaged_iZ_W'+str(theta_T_arcmins)+'_alpha_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'.dat', dat.T)
    else:
        dat = np.array([bincenters, binedges[:-1], binedges[1:], A2pt])
        np.savetxt(A2pt_path+'A2pt_iZ_W'+str(theta_T_arcmins)+'_alpha_'+str(int(min_sep_tc))+'_'+str(int(max_sep_tc))+'_'+str(nbins_tc)+'.dat', dat.T)
    
    print(f'Computing the area prefactors for iZ took {(time.time()-area_compute_time):.2f}s', flush=True)

if (compute_H_chi_D == 'yes'):

    H_chi_D_path = input.H_chi_D_path
    if (os.path.isdir(H_chi_D_path) == False):
        os.mkdir(H_chi_D_path)

#####################

def main_function():

    pool_opened = False
    if (compute_P_grid == 'yes' or compute_B_grid == 'yes' or compute_iB_grid == 'yes'):
        pool_opened = True
        # Open pool of workers to evaluate multiple points on the (l,z) grid simultaneously
        pool = mp.Pool(processes=constants._num_of_prcoesses_)
        print('\nOpened a pool of', constants._num_of_prcoesses_, 'worker processors', flush=True)

    for param_idx in range(start_idx, stop_idx):

        filename_extension = '_param_idx_' + str(param_idx) + '.dat'

        print('\n##############################', flush=True)

        #################################################################################################################################
        #################################################################################################################################
        # STEP 1: Create class object and helper cosmology object
        #################################################################################################################################
        #################################################################################################################################

        cosmo_pars_fid = input.cosmo_pars_fid
        params_dict = input.params_dict

        print('\nSetting up parameters for node index: '+str(param_idx)+'\n', flush=True)

        if ('Omega_b' in params_dict.keys()):
            Omega_b = params_dict['Omega_b'][param_idx]
            print('Omega_b SET to', Omega_b, flush=True)
        else:
            Omega_b = cosmo_pars_fid['Omega_b']
            print('Omega_b FIXED to fiducial value of', Omega_b, flush=True)
        
        if ('Omega_m' in params_dict.keys()):
            Omega_m = params_dict['Omega_m'][param_idx]
            print('Omega_m SET to', Omega_m, flush=True)
        else:
            Omega_m = cosmo_pars_fid['Omega_m'] 
            print('Omega_m FIXED to fiducial value of', Omega_m, flush=True)

        if ('h' in params_dict.keys()):
            h = params_dict['h'][param_idx]
            print('h SET to', h, flush=True)
        else:
            h = cosmo_pars_fid['h']
            print('h FIXED to fiducial value of', h, flush=True)

        if ('A_s' in params_dict.keys()):
            A_s = params_dict['A_s'][param_idx]
            sigma8_or_A_s = 'A_s'
            print('A_s SET to', A_s, flush=True)
        elif ('sigma8' in params_dict.keys()):
            sigma8 = params_dict['sigma8'][param_idx]
            sigma8_or_A_s = 'sigma8'
            print('sigma8 SET to', sigma8, flush=True)
        else:
            if ('A_s' in cosmo_pars_fid.keys()):
                A_s = cosmo_pars_fid['A_s']
                sigma8_or_A_s = 'A_s'
                print('A_s FIXED to fiducial value of', A_s, flush=True)
            elif ('sigma8' in cosmo_pars_fid.keys()):
                sigma8 = cosmo_pars_fid['sigma8']
                sigma8_or_A_s = 'sigma8'
                print('sigma8 FIXED to fiducial value of', sigma8, flush=True)

        if ('n_s' in params_dict.keys()):
            n_s = params_dict['n_s'][param_idx]
            print('n_s SET to', n_s, flush=True)
        else:
            n_s = cosmo_pars_fid['n_s']
            print('n_s FIXED to fiducial value of', n_s, flush=True)

        if ('w0' in params_dict.keys()):
            w0 = params_dict['w0'][param_idx]
            print('w0 SET to', w0, flush=True)
        else:
            w0 = cosmo_pars_fid['w0']
            print('w0 FIXED to fiducial value of', w0, flush=True)

        if ('wa' in params_dict.keys()):
            wa = params_dict['wa'][param_idx]
            print('wa SET to', wa, flush=True)
        else:
            wa = cosmo_pars_fid['wa']
            print('wa FIXED to fiducial value of', wa, flush=True)

        if ('c_min' in params_dict.keys()):
            c_min = params_dict['c_min'][param_idx]
            eta_0 = eta_0_val(c_min)
            print('c_min SET to', c_min, flush=True)
            print('eta_0 SET to', eta_0, flush=True)
        else:
            c_min = cosmo_pars_fid['c_min']
            eta_0 = cosmo_pars_fid['eta_0']
            print('c_min FIXED to fiducial value of', c_min, flush=True)
            print('eta_0 FIXED to fiducial value of', eta_0, flush=True)

        if ('Mv' in params_dict.keys()):
            Mv = params_dict['Mv'][param_idx]
            print('Mv SET to', Mv, flush=True)
        else:
            Mv = cosmo_pars_fid['Mv']
            print('Mv FIXED to fiducial value of', Mv, flush=True)

        if ('z' in params_dict.keys()):
            z_val = params_dict['z'][param_idx]
            z_array = np.array([params_dict['z'][param_idx]])
            print('z SET to', z_val, flush=True)
            print('z_array SET to', z_array, flush=True)
        else:
            z_val = float('NaN')
            z_array = input.z_array
            print('z FIXED to NaN', flush=True)
            print('z_array FIXED to default z-grid values', flush=True)

        if ('A_IA_NLA' in params_dict.keys()):
            A_IA_NLA = np.array([params_dict['A_IA_NLA'][param_idx]])
            print('A_IA_NLA SET to', A_IA_NLA, flush=True)
        else:
            A_IA_NLA = cosmo_pars_fid['A_IA_NLA']
            print('A_IA_NLA FIXED to fiducial value of', A_IA_NLA, flush=True)

        if ('alpha_IA_NLA' in params_dict.keys()):
            alpha_IA_NLA = np.array([params_dict['alpha_IA_NLA'][param_idx]])
            print('alpha_IA_NLA SET to', alpha_IA_NLA, flush=True)
        else:
            alpha_IA_NLA = cosmo_pars_fid['alpha_IA_NLA']
            print('alpha_IA_NLA FIXED to fiducial value of', alpha_IA_NLA, flush=True)

        class_settings  = {
                    'h':h,
                    'Omega_b':Omega_b,
                    'Omega_m':Omega_m,
                    'n_s':n_s,
                    'Omega_Lambda':0.0,
                    'fluid_equation_of_state':'CLP',
                    'w0_fld':w0,
                    'wa_fld':wa,
                    'output':'dTk,mPk',
                    'P_k_max_1/Mpc':constants._k_max_pk_,
                    'z_max_pk':constants._z_max_pk_,
                    'non linear':'hmcode',
                    'c_min':c_min,
                    'eta_0':eta_0,
                    }
        
        if (Mv != 0.0):
            # for massive neutrinos

            m1, m2, m3 = get_masses(Mv, input.NEUTRINO_HIERARCHY)
            
            print('Neutrino masses --> m1, m2, m3 in [eV] =', m1, m2, m3, flush=True)

            class_settings['N_ur'] = 0.00441
            class_settings['N_ncdm'] = 3
            class_settings['m_ncdm'] = str(m1)+','+str(m2)+','+str(m3)
            #class_settings['ncdm_fluid_approximation'] = 3 # will result in significantly slower running

        if (sigma8_or_A_s == 'A_s'):
            class_settings['A_s'] = A_s
        elif (sigma8_or_A_s == 'sigma8'):
            class_settings['sigma8'] = sigma8

        class_start = time.time()

        try:
            cclass = Class()
            cclass.set(class_settings)
            cclass.compute()
        except:
            print('!!! Encountered classy error in node index #'+str(param_idx)+' . Skipping it !!!', flush=True)
            continue

        if (sigma8_or_A_s == 'sigma8'):
            A_s = cclass.get_current_derived_parameters(['A_s'])['A_s']
            print('A_s COMPUTED to be', A_s, flush=True)
        elif (sigma8_or_A_s == 'A_s'):
            sigma8 = cclass.sigma8()
            print('sigma8 COMPUTED to be', sigma8, flush=True)

        S8 = cclass.S8()
        print('S8 COMPUTED to be', S8, flush=True)

        print('\nThe classy.CLASS object is made!', flush=True)

        CosmoClassObject = CosmoClass(cclass)

        class_end = time.time()
        print('The computation of classy.CLASS and CosmoClass objects took %ss\n'%(class_end - class_start), flush=True)

        cosmo_params_output_arr = np.array([Omega_b, Omega_m, h, A_s, sigma8, S8, n_s, w0, wa, c_min, eta_0, Mv, z_val, A_IA_NLA, alpha_IA_NLA])
        np.savetxt(cosmo_parameters_output_path+'cosmo_params'+filename_extension, cosmo_params_output_arr.T, header='Omega_b, Omega_m, h, A_s, sigma8, S8, n_s, w0, wa, c_min, eta_0, Mv, z_val, A_IA_NLA, alpha_IA_NLA')

        #################################################################################################################################
        #################################################################################################################################
        # STEP 2: Create (l,z) grids for P_3D(l,z), B_3D(l1,l2,l3,z) and/or iB_3D(l,z) (also for H_chi_D values)
        #################################################################################################################################
        #################################################################################################################################
        
        if (compute_H_chi_D == 'yes'):
            if (z_val == float('NaN')):
                print('Specify the redshift z at which to compute H(z), chi(z) and D(z) values!', flush=True)
            else:
                H_chi_D_vals = np.array([CosmoClassObject.H_z(z_val), CosmoClassObject.chi_z(z_val), CosmoClassObject.D_plus_z(z_val)])
                np.savetxt(H_chi_D_path+'H_chi_D'+filename_extension, H_chi_D_vals.T)

        if (compute_P_grid == 'yes'):

            print('Starting computation of P(l,z) grid', flush=True)
            start_grid = time.time()

            # Power spectrum

            P_l_z_param_0_array = l_array
            P_l_z_param_1_array = z_array
            P_l_z_param_2_array = [CosmoClassObject]
            P_l_z_param_3_array = [P3D_type]

            P_l_z_paramlist = list(itertools.product(P_l_z_param_0_array, P_l_z_param_1_array, P_l_z_param_2_array, P_l_z_param_3_array))

            print('Computing P(l,z) grid', flush=True)

            results_P_l_z = pool.map(P_l_z, P_l_z_paramlist)
            P_l_z_grid = np.array(results_P_l_z)
            P_l_z_grid = P_l_z_grid.reshape((l_array.size, z_array.size))

            np.savetxt(P_l_z_grid_path+'P_l_z_grid'+filename_extension, P_l_z_grid)

            # scale-dependent stochasticity
            #results_P_eps_eps_l_z = pool.map(P_eps_eps_l_z, P_l_z_paramlist)
            #P_eps_eps_l_z_grid = np.array(results_P_eps_eps_l_z)
            #P_eps_eps_l_z_grid = P_eps_eps_l_z_grid.reshape((l_array.size, z_array.size))

            #np.savetxt(P_l_z_grid_path+'P_eps_eps_l_z_grid'+filename_extension, P_eps_eps_l_z_grid)

            end_grid = time.time()
            print('Computing the P(l,z) grid took %ss'%(end_grid - start_grid), flush=True)

        if (compute_B_grid == 'yes'):

            print('Starting computation of B(l1,l2,l3,z) grid', flush=True)
            start_grid = time.time()

            # Bispectrum

            B_l1_l2_l3_z_param_0_array = l_B_array
            B_l1_l2_l3_z_param_1_array = l_B_array
            B_l1_l2_l3_z_param_2_array = l_B_array
            B_l1_l2_l3_z_param_3_array = z_B_array
            B_l1_l2_l3_z_param_4_array = [CosmoClassObject]
            B_l1_l2_l3_z_param_5_array = [B3D_type]

            B_l1_l2_3_z_paramlist = list(itertools.product(B_l1_l2_l3_z_param_0_array, B_l1_l2_l3_z_param_1_array, B_l1_l2_l3_z_param_2_array,
                                                           B_l1_l2_l3_z_param_3_array, B_l1_l2_l3_z_param_4_array, B_l1_l2_l3_z_param_5_array))

            print('Computing B(l1,l2,l3,z) grid', flush=True)

            results_B_l1_l2_l3_z = pool.map(B_l1_l2_l3_z, B_l1_l2_3_z_paramlist)
            B_l1_l2_l3_z_grid_flattened = np.array(results_B_l1_l2_l3_z)

            B_l1_l2_l3_z_grid = np.zeros((l_B_array.size, l_B_array.size, l_B_array.size, z_B_array.size))
            flattened_idx = 0

            for l1_idx in range(l_B_array.size):
                for l2_idx in range(l_B_array.size):
                    for l3_idx in range(l_B_array.size):
                        for z_idx in range(z_B_array.size):
                            B_l1_l2_l3_z_grid[l1_idx,l2_idx,l3_idx,z_idx] = B_l1_l2_l3_z_grid_flattened[flattened_idx]
                            flattened_idx += 1

            np.save(B_l1_l2_l3_z_grid_path+'B_l1_l2_l3_z_grid'+filename_extension, B_l1_l2_l3_z_grid) # save in .npy format

            end_grid = time.time()
            print('Computing the B(l1,l2,l3,z) grid took %ss'%(end_grid - start_grid), flush=True)

        if (compute_iB_grid == 'yes'):

            print('Starting computation of iB(l,z) grid', flush=True)
            start_grid = time.time()

            # Integrated bispectra

            iB_l_z_param_0_array = l_array
            iB_l_z_param_1_array = z_array
            iB_l_z_param_2_array = [theta_U]
            iB_l_z_param_3_array = [theta_T]
            iB_l_z_param_4_array = [CosmoClassObject]
            iB_l_z_param_5_array = [B3D_type]

            iB_l_z_paramlist = list(itertools.product(iB_l_z_param_0_array, iB_l_z_param_1_array, iB_l_z_param_2_array, iB_l_z_param_3_array, iB_l_z_param_4_array, iB_l_z_param_5_array))
            #pool = mp.Pool(processes=mp.cpu_count()-2)

            if ('app' in iZ_correlation_name_list):

                # iB_app : < shear aperture mass inside theta_U compensated * position-dependent shear-shear plus 2PCF inside theta_T tophat >

                print('Computing iB_app(l,z) grid', flush=True)

                try:
                    results_iB_app_l_z = pool.map(iB_app_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_app_l_z = pool.map(iB_app_l_z_integration, iB_l_z_paramlist)

                iB_app_l_z_grid = np.array(results_iB_app_l_z) * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_app_l_z_grid = iB_app_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_app_l_z_grid'+filename_extension, iB_app_l_z_grid)

            if ('amm' in iZ_correlation_name_list):

                # iB_amm : < shear aperture mass inside theta_U compensated * position-dependent shear-shear minus 2PCF inside theta_T tophat >

                print('Computing iB_amm(l,z) grid', flush=True)

                try:
                    results_iB_amm_l_z = pool.map(iB_amm_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_amm_l_z = pool.map(iB_amm_l_z_integration, iB_l_z_paramlist)

                iB_amm_l_z_grid = np.array(results_iB_amm_l_z) * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_amm_l_z_grid = iB_amm_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_amm_l_z_grid'+filename_extension, iB_amm_l_z_grid)

            if ('att' in iZ_correlation_name_list):

                # iB_att : < shear aperture mass inside theta_U compensated * position-dependent tangential-shear 2PCF inside theta_T tophat >

                print('Computing iB_att(l,z) grid', flush=True)

                try:
                    results_iB_att_l_z = pool.map(iB_att_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_att_l_z = pool.map(iB_att_l_z_integration, iB_l_z_paramlist)    

                results_iB_att_l_z = np.array(results_iB_att_l_z)

                # Term in iB_att which has the coefficient b1
                iB_att_b1_l_z_grid = results_iB_att_l_z[:,0] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_att_b1_l_z_grid = iB_att_b1_l_z_grid.reshape((l_array.size, z_array.size)) 

                # Term in iB_att which has the coefficient b2
                iB_att_b2_l_z_grid = results_iB_att_l_z[:,1] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_att_b2_l_z_grid = iB_att_b2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_att which has the coefficient bs2
                iB_att_bs2_l_z_grid = results_iB_att_l_z[:,2] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_att_bs2_l_z_grid = iB_att_bs2_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_att_b1_l_z_grid'+filename_extension, iB_att_b1_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_att_b2_l_z_grid'+filename_extension, iB_att_b2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_att_bs2_l_z_grid'+filename_extension, iB_att_bs2_l_z_grid)

            if ('agg' in iZ_correlation_name_list):

                # iB_agg : < shear aperture mass inside theta_U compensated * position-dependent galaxy clustering 2PCF inside theta_T tophat >

                print('Computing iB_agg(l,z) grid', flush=True)

                try:
                    results_iB_agg_l_z = pool.map(iB_agg_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_agg_l_z = pool.map(iB_agg_l_z_integration, iB_l_z_paramlist)

                results_iB_agg_l_z = np.array(results_iB_agg_l_z)

                # Term in iB_agg which has the coefficient b1^2
                iB_agg_b1sq_l_z_grid = results_iB_agg_l_z[:,0] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_agg_b1sq_l_z_grid = iB_agg_b1sq_l_z_grid.reshape((l_array.size, z_array.size)) # b1 term

                # Term in iB_agg which has the coefficient b1*b2
                iB_agg_b1_b2_l_z_grid = results_iB_agg_l_z[:,1] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_agg_b1_b2_l_z_grid = iB_agg_b1_b2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_agg which has the coefficient b1*bs2
                iB_agg_b1_bs2_l_z_grid = results_iB_agg_l_z[:,2] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_agg_b1_bs2_l_z_grid = iB_agg_b1_bs2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_agg which captures the eps*eps_delta shot-noise
                iB_agg_eps_epsdelta_l_z_grid = results_iB_agg_l_z[:,3] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_agg_eps_epsdelta_l_z_grid = iB_agg_eps_epsdelta_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_agg_b1sq_l_z_grid'+filename_extension, iB_agg_b1sq_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_agg_b1_b2_l_z_grid'+filename_extension, iB_agg_b1_b2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_agg_b1_bs2_l_z_grid'+filename_extension, iB_agg_b1_bs2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_agg_eps_epsdelta_l_z_grid'+filename_extension, iB_agg_eps_epsdelta_l_z_grid)

            if ('gpp' in iZ_correlation_name_list):

                # iB_gpp : < mean galaxy density inside theta_T tophat * position-dependent shear-shear plus 2PCF inside theta_T tophat >

                print('Computing iB_gpp(l,z) grid', flush=True)

                try:
                    results_iB_gpp_l_z = pool.map(iB_gpp_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_gpp_l_z = pool.map(iB_gpp_l_z_integration, iB_l_z_paramlist)

                results_iB_gpp_l_z = np.array(results_iB_gpp_l_z)

                # Term in iB_gpp which has the coefficient b1
                iB_gpp_b1_l_z_grid = results_iB_gpp_l_z[:,0] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gpp_b1_l_z_grid = iB_gpp_b1_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_gpp which has the coefficient b2
                iB_gpp_b2_l_z_grid = results_iB_gpp_l_z[:,1] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gpp_b2_l_z_grid = iB_gpp_b2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_gpp which has the coefficient bs2
                iB_gpp_bs2_l_z_grid = results_iB_gpp_l_z[:,2] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gpp_bs2_l_z_grid = iB_gpp_bs2_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_gpp_b1_l_z_grid'+filename_extension, iB_gpp_b1_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_gpp_b2_l_z_grid'+filename_extension, iB_gpp_b2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_gpp_bs2_l_z_grid'+filename_extension, iB_gpp_bs2_l_z_grid)

            if ('gmm' in iZ_correlation_name_list):

                # iB_gmm : < mean galaxy density inside theta_T tophat * position-dependent shear-shear minus 2PCF inside theta_T tophat >

                print('Computing iB_gmm(l,z) grid', flush=True)

                try:
                    results_iB_gmm_l_z = pool.map(iB_gmm_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_gmm_l_z = pool.map(iB_gmm_l_z_integration, iB_l_z_paramlist)

                results_iB_gmm_l_z = np.array(results_iB_gmm_l_z)

                # Term in iB_gmm which has the coefficient b1
                iB_gmm_b1_l_z_grid = results_iB_gmm_l_z[:,0] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gmm_b1_l_z_grid = iB_gmm_b1_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_gmm which has the coefficient b2
                iB_gmm_b2_l_z_grid = results_iB_gmm_l_z[:,1] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gmm_b2_l_z_grid = iB_gmm_b2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_gmm which has the coefficient bs2
                iB_gmm_bs2_l_z_grid = results_iB_gmm_l_z[:,2] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gmm_bs2_l_z_grid = iB_gmm_bs2_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_gmm_b1_l_z_grid'+filename_extension, iB_gmm_b1_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_gmm_b2_l_z_grid'+filename_extension, iB_gmm_b2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_gmm_bs2_l_z_grid'+filename_extension, iB_gmm_bs2_l_z_grid)

            if ('gtt' in iZ_correlation_name_list):

                # iB_gtt : < mean galaxy density inside theta_T tophat * position-dependent tangential-shear 2PCF inside theta_T tophat >

                print('Computing iB_gtt(l,z) grid', flush=True)

                try:
                    results_iB_gtt_l_z = pool.map(iB_gtt_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_gtt_l_z = pool.map(iB_gtt_l_z_integration, iB_l_z_paramlist)
                
                results_iB_gtt_l_z = np.array(results_iB_gtt_l_z)

                # Term in iB_gtt which has the coefficient b1^2
                iB_gtt_b1sq_l_z_grid = results_iB_gtt_l_z[:,0] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gtt_b1sq_l_z_grid = iB_gtt_b1sq_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_gtt which has the coefficient b1*b2
                iB_gtt_b1_b2_l_z_grid = results_iB_gtt_l_z[:,1] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gtt_b1_b2_l_z_grid = iB_gtt_b1_b2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_gtt which has the coefficient b1*bs2
                iB_gtt_b1_bs2_l_z_grid = results_iB_gtt_l_z[:,2] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gtt_b1_bs2_l_z_grid = iB_gtt_b1_bs2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_gtt which captures the eps*eps_delta shot-noise
                iB_gtt_eps_epsdelta_l_z_grid = results_iB_gtt_l_z[:,3] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gtt_eps_epsdelta_l_z_grid = iB_gtt_eps_epsdelta_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_gtt_b1sq_l_z_grid'+filename_extension, iB_gtt_b1sq_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_gtt_b1_b2_l_z_grid'+filename_extension, iB_gtt_b1_b2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_gtt_b1_bs2_l_z_grid'+filename_extension, iB_gtt_b1_bs2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_gtt_eps_epsdelta_l_z_grid'+filename_extension, iB_gtt_eps_epsdelta_l_z_grid)

            if ('ggg' in iZ_correlation_name_list):

                # iB_ggg : < mean galaxy density inside theta_T tophat * position-dependent galaxy clustering 2PCF inside theta_T tophat >

                print('Computing iB_ggg(l,z) grid', flush=True)

                try:
                    results_iB_ggg_l_z = pool.map(iB_ggg_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_ggg_l_z = pool.map(iB_ggg_l_z_integration, iB_l_z_paramlist)

                results_iB_ggg_l_z = np.array(results_iB_ggg_l_z)

                # Term in iB_ggg which has the coefficient b1^3
                iB_ggg_b1cu_l_z_grid = results_iB_ggg_l_z[:,0] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_ggg_b1cu_l_z_grid = iB_ggg_b1cu_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_ggg which has the coefficient b1^2*b2
                iB_ggg_b1sq_b2_l_z_grid = results_iB_ggg_l_z[:,1] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_ggg_b1sq_b2_l_z_grid = iB_ggg_b1sq_b2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_ggg which has the coefficient b1^2*bs2
                iB_ggg_b1sq_bs2_l_z_grid = results_iB_ggg_l_z[:,2] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_ggg_b1sq_bs2_l_z_grid = iB_ggg_b1sq_bs2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_ggg which captures the eps*eps_delta shot-noise and has the coefficient b1
                iB_ggg_b1_eps_epsdelta_l_z_grid = results_iB_ggg_l_z[:,3] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_ggg_b1_eps_epsdelta_l_z_grid = iB_ggg_b1_eps_epsdelta_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_ggg which captures the eps*eps*eps shot-noise
                iB_ggg_eps_eps_eps_l_z_grid = results_iB_ggg_l_z[:,4] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_ggg_eps_eps_eps_l_z_grid = iB_ggg_eps_eps_eps_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_ggg_b1cu_l_z_grid'+filename_extension, iB_ggg_b1cu_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_ggg_b1sq_b2_l_z_grid'+filename_extension, iB_ggg_b1sq_b2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_ggg_b1sq_bs2_l_z_grid'+filename_extension, iB_ggg_b1sq_bs2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_ggg_b1_eps_epsdelta_l_z_grid'+filename_extension, iB_ggg_b1_eps_epsdelta_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_ggg_eps_eps_eps_l_z_grid'+filename_extension, iB_ggg_eps_eps_eps_l_z_grid)

            if ('akk' in iZ_correlation_name_list):

                # iB_akk : < mean convergence inside theta_T tophat * position-dependent convergence 2PCF inside theta_T tophat >

                print('Computing iB_akk(l,z) grid', flush=True)

                try:
                    results_iB_akk_l_z = pool.map(iB_akk_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_akk_l_z = pool.map(iB_akk_l_z_integration, iB_l_z_paramlist)

                iB_akk_l_z_grid = np.array(results_iB_akk_l_z) * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_akk_l_z_grid = iB_akk_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_akk_l_z_grid'+filename_extension, iB_akk_l_z_grid)

            if ('agk' in iZ_correlation_name_list):

                # iB_agk : < mean convergence inside theta_T tophat * position-dependent galaxy-convergence 2PCF inside theta_T tophat >

                print('Computing iB_agk(l,z) grid', flush=True)

                try:
                    results_iB_agk_l_z = pool.map(iB_agk_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_agk_l_z = pool.map(iB_agk_l_z_integration, iB_l_z_paramlist)

                results_iB_agk_l_z = np.array(results_iB_agk_l_z)

                # Term in iB_agk which has the coefficient b1
                iB_agk_b1_l_z_grid = results_iB_agk_l_z[:,0] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_agk_b1_l_z_grid = iB_agk_b1_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_agk which has the coefficient b2
                iB_agk_b2_l_z_grid = results_iB_agk_l_z[:,1] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_agk_b2_l_z_grid = iB_agk_b2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_agk which has the coefficient bs2
                iB_agk_bs2_l_z_grid = results_iB_agk_l_z[:,2] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_agk_bs2_l_z_grid = iB_agk_bs2_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_agk_b1_l_z_grid'+filename_extension, iB_agk_b1_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_agk_b2_l_z_grid'+filename_extension, iB_agk_b2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_agk_bs2_l_z_grid'+filename_extension, iB_agk_bs2_l_z_grid)

            if ('gkk' in iZ_correlation_name_list):

                # iB_gkk : < mean galaxy density inside theta_T tophat * position-dependent convergence 2PCF inside theta_T tophat >

                print('Computing iB_gkk(l,z) grid', flush=True)

                try:
                    results_iB_gkk_l_z = pool.map(iB_gkk_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_gkk_l_z = pool.map(iB_gkk_l_z_integration, iB_l_z_paramlist)

                results_iB_gkk_l_z = np.array(results_iB_gkk_l_z)

                # Term in iB_gkk which has the coefficient b1
                iB_gkk_b1_l_z_grid = results_iB_gkk_l_z[:,0] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gkk_b1_l_z_grid = iB_gkk_b1_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_gkk which has the coefficient b2
                iB_gkk_b2_l_z_grid = results_iB_gkk_l_z[:,1] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gkk_b2_l_z_grid = iB_gkk_b2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_gkk which has the coefficient bs2
                iB_gkk_bs2_l_z_grid = results_iB_gkk_l_z[:,2] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_gkk_bs2_l_z_grid = iB_gkk_bs2_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_gkk_b1_l_z_grid'+filename_extension, iB_gkk_b1_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_gkk_b2_l_z_grid'+filename_extension, iB_gkk_b2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_gkk_bs2_l_z_grid'+filename_extension, iB_gkk_bs2_l_z_grid)

            if ('ggk' in iZ_correlation_name_list):

                # iB_ggk : < mean galaxy density inside theta_T tophat * position-dependent galaxy-convergence 2PCF inside theta_T tophat >

                print('Computing iB_ggk(l,z) grid', flush=True)

                try:
                    results_iB_ggk_l_z = pool.map(iB_ggk_l_z_integration, iB_l_z_paramlist)
                except: # for cases when the first trial fails (for whatever reason), just giving it one more try
                    results_iB_ggk_l_z = pool.map(iB_ggk_l_z_integration, iB_l_z_paramlist)

                results_iB_ggk_l_z = np.array(results_iB_ggk_l_z)

                # Term in iB_ggk which has the coefficient b1^2
                iB_ggk_b1sq_l_z_grid = results_iB_ggk_l_z[:,0] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_ggk_b1sq_l_z_grid = iB_ggk_b1sq_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_ggk which has the coefficient b1*b2
                iB_ggk_b1_b2_l_z_grid = results_iB_ggk_l_z[:,1] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_ggk_b1_b2_l_z_grid = iB_ggk_b1_b2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_ggk which has the coefficient b1*bs2
                iB_ggk_b1_bs2_l_z_grid = results_iB_ggk_l_z[:,2] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_ggk_b1_bs2_l_z_grid = iB_ggk_b1_bs2_l_z_grid.reshape((l_array.size, z_array.size))

                # Term in iB_ggk which captures the eps*eps_delta shot-noise
                iB_ggk_eps_epsdelta_l_z_grid = results_iB_ggk_l_z[:,3] * (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)
                iB_ggk_eps_epsdelta_l_z_grid = iB_ggk_eps_epsdelta_l_z_grid.reshape((l_array.size, z_array.size))

                np.savetxt(iB_l_z_grid_path+'iB_ggk_b1sq_l_z_grid'+filename_extension, iB_ggk_b1sq_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_ggk_b1_b2_l_z_grid'+filename_extension, iB_ggk_b1_b2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_ggk_b1_bs2_l_z_grid'+filename_extension, iB_ggk_b1_bs2_l_z_grid)
                np.savetxt(iB_l_z_grid_path+'iB_ggk_eps_epsdelta_l_z_grid'+filename_extension, iB_ggk_eps_epsdelta_l_z_grid)

            end_grid = time.time()
            print('Computing the iB(l,z) grids took %ss'%(end_grid - start_grid), flush=True)

        #################################################################################################################################
        #################################################################################################################################
        # STEP 3: Peform line-of-sight projection of the (l,z) grids to obtain spectra and Fourier transform them to get correlations
        #################################################################################################################################
        #################################################################################################################################

        if (compute_B_spectra == 'yes'):

            print('Starting computation of B(l1,l2,l3) spectra', flush=True)
            start_spectra_and_correlations = time.time()

            if (compute_B_spectra == 'yes' and compute_B_grid == 'no'):
                
                B_l1_l2_l3_z_grid = np.load(B_l1_l2_l3_z_grid_path+'B_l1_l2_l3_z_grid'+filename_extension+'.npy') # load .npy format

            #####################
            # setup LOS projection kernels

            H_0 = CosmoClassObject.H_z(0)
            Omega0_m = CosmoClassObject.Omega0_m

            z_array_los = np.linspace(0.01, constants._z_max_, constants._N_los_)
            chi_inv_z_array_los = 1./CosmoClassObject.chi_z(z_array_los)
            H_inv_z_array_los = 1./CosmoClassObject.H_z(z_array_los)

            # ----------
            # source bins

            qs_z_array_los = np.zeros([len(SOURCE_BIN_NAME_LIST), z_array_los.size])

            for SOURCE_BIN_idx in range(len(SOURCE_BIN_NAME_LIST)):
                SOURCE_BIN_NAME = SOURCE_BIN_NAME_LIST[SOURCE_BIN_idx]

                if ('zs' in SOURCE_BIN_NAME):
                    z_source = SOURCE_BIN_VALUES[SOURCE_BIN_idx]
                elif ('BIN' in SOURCE_BIN_NAME):
                    if (input.use_Dirac_comb == True):
                        # for Dirac comb
                        n_s_z_BIN_z_tab = np.loadtxt('./../data/nofz/DESY3_nofz/Dirac_comb/nofz_DESY3_source_'+SOURCE_BIN_NAME+'_Dirac_center.tab', usecols=[0])
                        n_s_z_BIN_vals_tab = np.loadtxt('./../data/nofz/DESY3_nofz/Dirac_comb/nofz_DESY3_source_'+SOURCE_BIN_NAME+'_Dirac_center.tab', usecols=[1])
                        print(str(SOURCE_BIN_NAME)+' '+str(np.sum(n_s_z_BIN_vals_tab)), flush=True)

                    else:
                        n_s_z_BIN_z_tab, n_s_z_BIN_vals_tab = np.loadtxt('./../data/nofz/DESY3_nofz/nofz_DESY3_source_'+SOURCE_BIN_NAME+'.tab').T
                        n_s_z_BIN_vals_tab /= np.trapezoid(n_s_z_BIN_vals_tab, n_s_z_BIN_z_tab)
                        n_s_z_BIN = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN_vals_tab, fill_value=(0,0), bounds_error=False)
                
                for j in range(z_array_los.size):
                    if ('zs' in SOURCE_BIN_NAME):
                        qs_z_array_los[SOURCE_BIN_idx,j] = q_k_zs_fixed(z_array_los[j], z_source, CosmoClassObject.chi_z, H_0, Omega0_m)
                    elif ('BIN' in SOURCE_BIN_NAME):

                        if (input.use_Dirac_comb == True):
                            # for Dirac comb
                            weights_sum = np.sum(n_s_z_BIN_vals_tab)
                            for zs_plane_idx in range(n_s_z_BIN_z_tab.size):
                                qs_z_array_los[SOURCE_BIN_idx,j] += n_s_z_BIN_vals_tab[zs_plane_idx]*q_k_zs_fixed(z_array_los[j], n_s_z_BIN_z_tab[zs_plane_idx], CosmoClassObject.chi_z, H_0, Omega0_m)

                            qs_z_array_los[SOURCE_BIN_idx,j] /= weights_sum
                    
                        else:
                            #qs_z_array_los[SOURCE_BIN_idx,j] = q_k_zs_distribution(z_array_los[j], n_s_z_BIN_z_tab[-1], CosmoClassObject.chi_z, H_0, Omega0_m, n_s_z_BIN)
                            qs_z_array_los[SOURCE_BIN_idx,j] = q_k_zs_distribution_systematics(z_array_los[j], n_s_z_BIN_z_tab[-1], CosmoClassObject.chi_z, H_0, Omega0_m, n_s_z_BIN, CosmoClassObject.H_z, CosmoClassObject.D_plus_z, A_IA_NLA, alpha_IA_NLA, SOURCE_BIN_delta_photoz_values[SOURCE_BIN_idx], SOURCE_BIN_m_values[SOURCE_BIN_idx])

            #####################
            # compute spectra

            print('Computing B(l1,l2,l3) spectra', flush=True)

            corr_idx = 0
            for a in range(len(SOURCE_BIN_NAME_LIST)):
                for b in range(a, len(SOURCE_BIN_NAME_LIST)):
                    for c in range(b, len(SOURCE_BIN_NAME_LIST)):
                        assert(corr_idx != num_i3pt_sss_correlations)

                        q1_z_array_los = np.zeros(z_array_los.size)
                        q2_z_array_los = np.zeros(z_array_los.size)
                        q3_z_array_los = np.zeros(z_array_los.size)

                        q1_bin_name = SOURCE_BIN_NAME_LIST[a]
                        q2_bin_name = SOURCE_BIN_NAME_LIST[b]
                        q3_bin_name = SOURCE_BIN_NAME_LIST[c]

                        q1_z_array_los = qs_z_array_los[a]
                        q2_z_array_los = qs_z_array_los[b]
                        q3_z_array_los = qs_z_array_los[c]

                        B_z_weight_array_los = H_inv_z_array_los * chi_inv_z_array_los**4 * q1_z_array_los * q2_z_array_los * q3_z_array_los # bispectrum LOS weighting

                        B_l1_l2_l3_z_grid_los = np.zeros([l_B_array.size, l_B_array.size, l_B_array.size, z_array_los.size])

                        for l1_idx in range(l_B_array.size):
                            for l2_idx in range(l_B_array.size):
                                for l3_idx in range(l_B_array.size):
                                    B_1_l1_l2_l3_z_func = interpolate.interp1d(z_B_array, B_l1_l2_l3_z_grid[l1_idx,l2_idx,l3_idx], fill_value=(0,0), bounds_error=False)
                                    B_l1_l2_l3_z_grid_los[l1_idx,l2_idx,l3_idx] = B_1_l1_l2_l3_z_func(z_array_los)
                                                
                        B_l1_l2_l3 = np.zeros([l_B_array.size, l_B_array.size, l_B_array.size])

                        B_1_l1_l2_l3_z_integrand_grid = B_l1_l2_l3_z_grid_los * B_z_weight_array_los
                        B_l1_l2_l3 = np.trapezoid(B_1_l1_l2_l3_z_integrand_grid, x=z_array_los, axis=-1)

                        np.save(B_spectra_path+'B_l1_l2_l3_'+q1_bin_name+'_'+q2_bin_name+'_'+q3_bin_name+filename_extension, B_l1_l2_l3)

                        # TODO: Still to implement B(ell_1, ell_2, ell_3) with Kitching correction and interpolation
                        '''
                        ell, iB_1_ell = C_ell_spherical_sky(l_array, iB_l[0])
                        ell, iB_2_ell = C_ell_spherical_sky(l_array, iB_l[1])
                        ell, iB_3_ell = C_ell_spherical_sky(l_array, iB_l[2])
                        ell, iB_4_ell = C_ell_spherical_sky(l_array, iB_l[3])
                        ell, iB_5_ell = C_ell_spherical_sky(l_array, iB_l[4])

                        #if nside is given, consider the pixel window function's effect on the C_ells
                        if nside is not None and healpy_installed:
                            
                            pixwin = hp.pixwin(nside, lmax=np.max(ell))
                            pixwin_ell = np.arange(len(pixwin))                                         # pixwin creates window function for ell=0 to 3*nside-1
                            pixwin = pixwin[np.intersect1d(ell, pixwin_ell, return_indices=True)[2]]    # match pixwin function to correct ells
                            pixwin = np.append(pixwin, np.zeros(len(ell) - len(pixwin)))                # Add zeros to match length
                            
                            # Add correction to C_ell
                            iB_1_ell = iB_1_ell * pixwin**2             
                            iB_2_ell = iB_2_ell * pixwin**2
                            iB_3_ell = iB_3_ell * pixwin**2
                            iB_4_ell = iB_4_ell * pixwin**2
                            iB_5_ell = iB_5_ell * pixwin**2
                            

                        iB_4_ell = iB_4_ell - np.mean(iB_4_ell[ell > 1000]) # for the shot noise terms
                        iB_5_ell = iB_5_ell - np.mean(iB_5_ell[ell > 1000]) # for the shot noise terms
                        '''
                           
                        corr_idx += 1

            end_spectra_and_correlations = time.time()
            print('Computing the B(l1,l2,l3) spectra took %ss'%(end_spectra_and_correlations - start_spectra_and_correlations), flush=True) 

        if (compute_P_spectra_and_correlations == 'yes' or compute_iB_spectra_and_correlations == 'yes'):

            print('Starting computation of P(l) and/or iB(l) spectra and xi(alpha) and/or iZ(alpha) correlations', flush=True)
            start_spectra_and_correlations = time.time()

            filename_extension_grid = filename_extension
            if (('delta_photoz' in filename_extension) or ('A_IA_NLA' in filename_extension) or ('alpha_IA_NLA' in filename_extension)):
                filename_extension_grid = '.dat'

            #####################
            # Load pre-computed grids

            if (compute_P_spectra_and_correlations == 'yes' and compute_P_grid == 'no'):
                
                P_l_z_grid = np.loadtxt(P_l_z_grid_path+'P_l_z_grid'+filename_extension_grid)
                
                # scale-dependent stochasticity
                #P_eps_eps_l_z_grid = np.loadtxt(P_l_z_grid_path+'P_eps_eps_l_z_grid'+filename_extension_grid)
                
            if (compute_iB_spectra_and_correlations == 'yes' and compute_iB_grid == 'no'):
                # For galaxy x shear (UWW) integrated bispectra grids

                if ('app' in iZ_correlation_name_list):
                    iB_app_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_app_l_z_grid'+filename_extension_grid)

                if ('amm' in iZ_correlation_name_list):
                    iB_amm_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_amm_l_z_grid'+filename_extension_grid)

                if ('att' in iZ_correlation_name_list):
                    iB_att_b1_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_att_b1_l_z_grid'+filename_extension_grid)
                    iB_att_b2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_att_b2_l_z_grid'+filename_extension_grid)
                    iB_att_bs2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_att_bs2_l_z_grid'+filename_extension_grid)

                if ('agg' in iZ_correlation_name_list):
                    iB_agg_b1sq_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_agg_b1sq_l_z_grid'+filename_extension_grid)
                    iB_agg_b1_b2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_agg_b1_b2_l_z_grid'+filename_extension_grid)
                    iB_agg_b1_bs2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_agg_b1_bs2_l_z_grid'+filename_extension_grid)
                    iB_agg_eps_epsdelta_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_agg_eps_epsdelta_l_z_grid'+filename_extension_grid)

                if ('gpp' in iZ_correlation_name_list):
                    iB_gpp_b1_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gpp_b1_l_z_grid'+filename_extension_grid)
                    iB_gpp_b2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gpp_b2_l_z_grid'+filename_extension_grid)
                    iB_gpp_bs2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gpp_bs2_l_z_grid'+filename_extension_grid)

                if ('gmm' in iZ_correlation_name_list):
                    iB_gmm_b1_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gmm_b1_l_z_grid'+filename_extension_grid)
                    iB_gmm_b2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gmm_b2_l_z_grid'+filename_extension_grid)
                    iB_gmm_bs2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gmm_bs2_l_z_grid'+filename_extension_grid)

                if ('gtt' in iZ_correlation_name_list):
                    iB_gtt_b1sq_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gtt_b1sq_l_z_grid'+filename_extension_grid)
                    iB_gtt_b1_b2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gtt_b1_b2_l_z_grid'+filename_extension_grid)
                    iB_gtt_b1_bs2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gtt_b1_bs2_l_z_grid'+filename_extension_grid)
                    iB_gtt_eps_epsdelta_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gtt_eps_epsdelta_l_z_grid'+filename_extension_grid)

                if ('ggg' in iZ_correlation_name_list):
                    iB_ggg_b1cu_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_ggg_b1cu_l_z_grid'+filename_extension_grid)
                    iB_ggg_b1sq_b2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_ggg_b1sq_b2_l_z_grid'+filename_extension_grid)
                    iB_ggg_b1sq_bs2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_ggg_b1sq_bs2_l_z_grid'+filename_extension_grid)
                    iB_ggg_b1_eps_epsdelta_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_ggg_b1_eps_epsdelta_l_z_grid'+filename_extension_grid)
                    iB_ggg_eps_eps_eps_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_ggg_eps_eps_eps_l_z_grid'+filename_extension_grid)

                if ('akk' in iZ_correlation_name_list):
                    iB_akk_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_akk_l_z_grid'+filename_extension_grid)

                if ('agk' in iZ_correlation_name_list):
                    iB_agk_b1_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_agk_b1_l_z_grid'+filename_extension_grid)
                    iB_agk_b2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_agk_b2_l_z_grid'+filename_extension_grid)
                    iB_agk_bs2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_agk_bs2_l_z_grid'+filename_extension_grid)

                if ('gkk' in iZ_correlation_name_list):
                    iB_gkk_b1_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gkk_b1_l_z_grid'+filename_extension_grid)
                    iB_gkk_b2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gkk_b2_l_z_grid'+filename_extension_grid)
                    iB_gkk_bs2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_gkk_bs2_l_z_grid'+filename_extension_grid)

                if ('ggk' in iZ_correlation_name_list):
                    iB_ggk_b1sq_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_ggk_b1sq_l_z_grid'+filename_extension_grid)
                    iB_ggk_b1_b2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_ggk_b1_b2_l_z_grid'+filename_extension_grid)
                    iB_ggk_b1_bs2_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_ggk_b1_bs2_l_z_grid'+filename_extension_grid)
                    iB_ggk_eps_epsdelta_l_z_grid = np.loadtxt(iB_l_z_grid_path+'iB_ggk_eps_epsdelta_l_z_grid'+filename_extension_grid)

            #####################
            # setup LOS projection kernels

            H_0 = CosmoClassObject.H_z(0)
            Omega0_m = CosmoClassObject.Omega0_m

            z_array_los = np.linspace(0.01, constants._z_max_, constants._N_los_)
            chi_inv_z_array_los = 1./CosmoClassObject.chi_z(z_array_los)
            H_inv_z_array_los = 1./CosmoClassObject.H_z(z_array_los)

            # ----------
            # source bins

            qs_z_array_los = np.zeros([len(SOURCE_BIN_NAME_LIST), z_array_los.size])

            for SOURCE_BIN_idx in range(len(SOURCE_BIN_NAME_LIST)):
                SOURCE_BIN_NAME = SOURCE_BIN_NAME_LIST[SOURCE_BIN_idx]

                if ('zs' in SOURCE_BIN_NAME):
                    z_source = SOURCE_BIN_VALUES[SOURCE_BIN_idx]
                elif ('BIN' in SOURCE_BIN_NAME):
                    if (input.use_Dirac_comb == True):
                        # for Dirac comb
                        n_s_z_BIN_z_tab = np.loadtxt('./../data/nofz/DESY3_nofz/Dirac_comb/nofz_DESY3_source_'+SOURCE_BIN_NAME+'_Dirac_center.tab', usecols=[0])
                        n_s_z_BIN_vals_tab = np.loadtxt('./../data/nofz/DESY3_nofz/Dirac_comb/nofz_DESY3_source_'+SOURCE_BIN_NAME+'_Dirac_center.tab', usecols=[1])
                        print(str(SOURCE_BIN_NAME)+' '+str(np.sum(n_s_z_BIN_vals_tab)), flush=True)

                    else:
                        n_s_z_BIN_z_tab, n_s_z_BIN_vals_tab = np.loadtxt('./../data/nofz/DESY3_nofz/nofz_DESY3_source_'+SOURCE_BIN_NAME+'.tab').T
                        n_s_z_BIN_vals_tab /= np.trapezoid(n_s_z_BIN_vals_tab, n_s_z_BIN_z_tab)
                        n_s_z_BIN = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN_vals_tab, fill_value=(0,0), bounds_error=False)
                
                for j in range(z_array_los.size):
                    if ('zs' in SOURCE_BIN_NAME):
                        qs_z_array_los[SOURCE_BIN_idx,j] = q_k_zs_fixed(z_array_los[j], z_source, CosmoClassObject.chi_z, H_0, Omega0_m)
                    elif ('BIN' in SOURCE_BIN_NAME):

                        if (input.use_Dirac_comb == True):
                            # for Dirac comb
                            weights_sum = np.sum(n_s_z_BIN_vals_tab)
                            for zs_plane_idx in range(n_s_z_BIN_z_tab.size):
                                qs_z_array_los[SOURCE_BIN_idx,j] += n_s_z_BIN_vals_tab[zs_plane_idx]*q_k_zs_fixed(z_array_los[j], n_s_z_BIN_z_tab[zs_plane_idx], CosmoClassObject.chi_z, H_0, Omega0_m)

                            qs_z_array_los[SOURCE_BIN_idx,j] /= weights_sum
                    
                        else:
                            #qs_z_array_los[SOURCE_BIN_idx,j] = q_k_zs_distribution(z_array_los[j], n_s_z_BIN_z_tab[-1], CosmoClassObject.chi_z, H_0, Omega0_m, n_s_z_BIN)
                            qs_z_array_los[SOURCE_BIN_idx,j] = q_k_zs_distribution_systematics(z_array_los[j], n_s_z_BIN_z_tab[-1], CosmoClassObject.chi_z, H_0, Omega0_m, n_s_z_BIN, CosmoClassObject.H_z, CosmoClassObject.D_plus_z, A_IA_NLA, alpha_IA_NLA, SOURCE_BIN_delta_photoz_values[SOURCE_BIN_idx], SOURCE_BIN_m_values[SOURCE_BIN_idx])

            if ('shear_x_shear' not in spectra_and_correlation_type):
                # Right now this is only for a single lens bin
                # ----------
                # lens bin

                if ('halos' not in LENS_BIN_NAME):
                    if ('redmagic' in input.LENS_BIN_FULLNAME_theory):
                        sample_name = 'redmagic'
                    elif ('maglim' in input.LENS_BIN_FULLNAME_theory):
                        sample_name = 'maglim'
                    n_l_z_BIN_z_tab = np.loadtxt('./../data/nofz/DESY3_nofz/nofz_DESY3_'+sample_name+'_lens_'+LENS_BIN_NAME+'.tab', usecols=[0])
                    n_l_z_BIN_vals_tab = np.loadtxt('./../data/nofz/DESY3_nofz/nofz_DESY3_'+sample_name+'_lens_'+LENS_BIN_NAME+'.tab', usecols=[1])
                    n_l_z_BIN = interpolate.interp1d(n_l_z_BIN_z_tab, n_l_z_BIN_vals_tab, fill_value=(0,0), bounds_error=False)
                    
                elif ('halos' in LENS_BIN_NAME):
                    if ('M' not in LENS_BIN_NAME):
                        # For halo bias (or bias that is non constant with redshift)
                        z_h_min = 0.1
                        z_h_max = 0.45
                        M_h_min = 10**12.0413926852 / CosmoClassObject.h
                        if ('halos2' in LENS_BIN_NAME):
                            M_h_min = 10**13.3424226808 / CosmoClassObject.h
                        M_h_max = 10**15.5 / CosmoClassObject.h
                        N_halos = N_h(z_h_min, z_h_max, M_h_min, M_h_max, CosmoClassObject.rho0_m, CosmoClassObject.sigma_R_z, CosmoClassObject.sigma_prime_R_z, CosmoClassObject.chi_z, CosmoClassObject.H_z)
                        print('Number of halos = ', N_halos, flush=True)
                    else:
                        n_l_z_BIN_z_tab = np.loadtxt('./../data/nofz/DESY3_nofz/nofz_mock_'+halo_type+'_'+LENS_BIN_NAME+'.tab', usecols=[0])
                        n_l_z_BIN_vals_tab = np.loadtxt('./../data/nofz/DESY3_nofz/nofz_mock_'+halo_type+'_'+LENS_BIN_NAME+'.tab', usecols=[1])
                        n_l_z_BIN = interpolate.interp1d(n_l_z_BIN_z_tab, n_l_z_BIN_vals_tab, fill_value=(0,0), bounds_error=False)
                        filename_extension = '_'+halo_type+'.dat'

                ql_b1_z_array_los = np.zeros(z_array_los.size)
                ql_b2_z_array_los = np.zeros(z_array_los.size)
                ql_bs2_z_array_los = np.zeros(z_array_los.size)
                ql_n_bar_z_array_los = np.zeros(z_array_los.size) # equivalent to q_g or q_eps
                ql_1_over_n_bar_z_array_los = np.zeros(z_array_los.size)

                for j in range(z_array_los.size):
                    if ('halos' in LENS_BIN_NAME and 'M' not in LENS_BIN_NAME):
                        ql_b1_z_array_los[j] = 1/N_halos * q_h_bias_unnormalised(z_array_los[j], z_h_min, z_h_max, M_h_min, M_h_max, 'b1', CosmoClassObject.rho0_m, CosmoClassObject.sigma_R_z, CosmoClassObject.sigma_prime_R_z, CosmoClassObject.chi_z)
                        ql_b2_z_array_los[j] = 1/N_halos * q_h_bias_unnormalised(z_array_los[j], z_h_min, z_h_max, M_h_min, M_h_max, 'b2', CosmoClassObject.rho0_m, CosmoClassObject.sigma_R_z, CosmoClassObject.sigma_prime_R_z, CosmoClassObject.chi_z)
                        ql_bs2_z_array_los[j] = 1/N_halos * q_h_bias_unnormalised(z_array_los[j], z_h_min, z_h_max, M_h_min, M_h_max, 'bs2', CosmoClassObject.rho0_m, CosmoClassObject.sigma_R_z, CosmoClassObject.sigma_prime_R_z, CosmoClassObject.chi_z)
                        ql_n_bar_z_array_los[j] = 1/N_halos * q_h_bias_unnormalised(z_array_los[j], z_h_min, z_h_max, M_h_min, M_h_max, 'n_h', CosmoClassObject.rho0_m, CosmoClassObject.sigma_R_z, CosmoClassObject.sigma_prime_R_z, CosmoClassObject.chi_z)
                        ql_1_over_n_bar_z_array_los[j] = 1/N_halos * q_h_bias_unnormalised(z_array_los[j], z_h_min, z_h_max, M_h_min, M_h_max, '1_over_n_h', CosmoClassObject.rho0_m, CosmoClassObject.sigma_R_z, CosmoClassObject.sigma_prime_R_z, CosmoClassObject.chi_z)
                    else :
                        val = q_m(z_array_los[j], n_l_z_BIN, CosmoClassObject.H_z)
                        ql_b1_z_array_los[j] = val
                        ql_b2_z_array_los[j] = val
                        ql_bs2_z_array_los[j] = val
                        ql_n_bar_z_array_los[j] = val
                        ql_1_over_n_bar_z_array_los[j] = val
                    
            #####################
            # compute spectra and correlations

            if (compute_P_spectra_and_correlations == 'yes'):

                print('Computing P(l) spectra and xi(alpha) correlations', flush=True)

                corr_idx = 0
                for a in range(len(SOURCE_BIN_NAME_LIST)):
                    for b in range(a, len(SOURCE_BIN_NAME_LIST)):
                        assert(corr_idx != num_2pt_sss_correlations)
                
                        for xi_correlation_name in xi_correlation_name_list:

                            q1_z_array_los = np.zeros([5, z_array_los.size])
                            q2_z_array_los = np.zeros([5, z_array_los.size])

                            if (xi_correlation_name == 'pp' or xi_correlation_name == 'mm' or xi_correlation_name == 'kk'):

                                q1_bin_name = SOURCE_BIN_NAME_LIST[a]
                                q2_bin_name = SOURCE_BIN_NAME_LIST[b]

                                q1_z_array_los[0,:] = qs_z_array_los[a]
                                q2_z_array_los[0,:] = qs_z_array_los[b]

                            elif (xi_correlation_name == 'tt' or xi_correlation_name == 'gk'):

                                q1_bin_name = LENS_BIN_NAME
                                q2_bin_name = SOURCE_BIN_NAME_LIST[b]

                                q1_z_array_los[0,:] = ql_b1_z_array_los
                                q2_z_array_los[0,:] = qs_z_array_los[b]

                            elif (xi_correlation_name == 'gg'):

                                q1_bin_name = LENS_BIN_NAME
                                q2_bin_name = LENS_BIN_NAME

                                q1_z_array_los[0,:] = ql_b1_z_array_los
                                q2_z_array_los[0,:] = ql_b1_z_array_los

                                # scale-dependent stochasticity
                                #q1_z_array_los[1,:] = ql_n_bar_z_array_los
                                #q2_z_array_los[1,:] = ql_1_over_n_bar_z_array_los

                            P_z_weight_array_los = H_inv_z_array_los * chi_inv_z_array_los**2 * q1_z_array_los * q2_z_array_los # Power spectrum LOS weighting
                            P_l_z_grid_los = np.zeros([l_array.size, z_array_los.size])

                            for i in range(l_array.size):
                                P_l_z_func = interpolate.interp1d(z_array, P_l_z_grid[i], fill_value=(0,0), bounds_error=False)
                                P_l_z_grid_los[i] = P_l_z_func(z_array_los)

                            P_l_z_integrand_grid = P_l_z_grid_los * P_z_weight_array_los[0,:]
                            P_l = np.trapezoid(P_l_z_integrand_grid, x=z_array_los, axis=1)

                            np.savetxt(P_spectra_path+'P_'+xi_correlation_name+'_l_'+q1_bin_name+'_'+q2_bin_name+filename_extension, P_l.T)

                            ell, P_ell = C_ell_spherical_sky(l_array, P_l)
                                                    
                            # If nside is given, calculate the pixel window function and correct the C_ell
                            if nside is not None and healpy_installed:
                                
                                pixwin = hp.pixwin(nside, lmax=np.max(ell))
                                pixwin_ell = np.arange(len(pixwin))                                         # pixwin creates window function for ell=0 to 3*nside-1
                                pixwin = pixwin[np.intersect1d(ell, pixwin_ell, return_indices=True)[2]]    # match pixwin function to correct ells
                                pixwin = np.append(pixwin, np.zeros(len(ell) - len(pixwin)))                # Add zeros to match length
                                
                                # Add correction to C_ell
                                P_ell = P_ell * pixwin**2
                                                        
                            xi_array_bin_averaged = np.zeros(alpha_min_arcmins_xi.size)
                            
                            if (xi_correlation_name == 'pp'):
                                xi_array_bin_averaged = xip_theta_bin_averaged(ell, P_ell)
                            elif (xi_correlation_name == 'mm'):
                                xi_array_bin_averaged = xim_theta_bin_averaged(ell, P_ell)
                            elif (xi_correlation_name == 'tt'):
                                xi_array_bin_averaged = xit_theta_bin_averaged(ell, P_ell)
                            elif (xi_correlation_name == 'gg' or xi_correlation_name == 'kk' or xi_correlation_name == 'gk'):
                                xi_array_bin_averaged = xi_theta_bin_averaged(ell, P_ell)   
                                
                            np.savetxt(xi_correlations_path+'xi_'+xi_correlation_name+'_'+q1_bin_name+'_'+q2_bin_name+filename_extension, xi_array_bin_averaged.T) 

                            '''
                            # scale-dependent stochasticity
                            if (xi_correlation_name == 'gg'):

                                P_eps_eps_l_z_grid_los = np.zeros([l_array.size, z_array_los.size])

                                for i in range(l_array.size):
                                    P_eps_eps_l_z_func = interpolate.interp1d(z_array, P_eps_eps_l_z_grid[i], fill_value=(0,0), bounds_error=False)
                                    P_eps_eps_l_z_grid_los[i] = P_eps_eps_l_z_func(z_array_los)

                                P_eps_eps_l_z_integrand_grid = P_eps_eps_l_z_grid_los * P_z_weight_array_los[1,:]
                                P_eps_eps_l = np.trapezoid(P_eps_eps_l_z_integrand_grid, x=z_array_los, axis=1)

                                np.savetxt(P_spectra_path+'P_eps_eps_'+xi_correlation_name+'_l_'+q1_bin_name+'_'+q2_bin_name+filename_extension, P_eps_eps_l.T)

                                ell, P_eps_eps_ell = C_ell_spherical_sky(l_array, P_eps_eps_l)

                                P_eps_eps_ell = P_eps_eps_ell - np.mean(P_eps_eps_ell[ell > 1000]) # for the shot noise terms

                                xi_eps_eps_array_bin_averaged = np.zeros(alpha_min_arcmins_xi.size)

                                for i in range(alpha_min_arcmins_xi.size):
                                    xi_eps_eps_array_bin_averaged[i] = xi_theta_bin_averaged(np.radians(alpha_min_arcmins_xi[i]/60), np.radians(alpha_max_arcmins_xi[i]/60), ell, P_eps_eps_ell)          

                                np.savetxt(xi_correlations_path+'xi_eps_eps_'+xi_correlation_name+'_'+q1_bin_name+'_'+q2_bin_name+filename_extension, xi_eps_eps_array_bin_averaged.T) 
                            '''
                            
                        corr_idx += 1

            if (compute_iB_spectra_and_correlations == 'yes'):     

                print('Computing iB(l) spectra and iZ(alpha) correlations', flush=True)

                corr_idx = 0
                for a in range(len(SOURCE_BIN_NAME_LIST)):
                    for b in range(a, len(SOURCE_BIN_NAME_LIST)):
                        for c in range(b, len(SOURCE_BIN_NAME_LIST)):
                            assert(corr_idx != num_i3pt_sss_correlations)

                            for iZ_correlation_name in iZ_correlation_name_list:

                                q1_z_array_los = np.zeros([5, z_array_los.size])
                                q2_z_array_los = np.zeros([5, z_array_los.size])
                                q3_z_array_los = np.zeros([5, z_array_los.size])

                                iB_1_l_z_grid = np.zeros([l_array.size, z_array.size])
                                iB_2_l_z_grid = np.zeros([l_array.size, z_array.size])
                                iB_3_l_z_grid = np.zeros([l_array.size, z_array.size])
                                iB_4_l_z_grid = np.zeros([l_array.size, z_array.size])
                                iB_5_l_z_grid = np.zeros([l_array.size, z_array.size])

                                if (iZ_correlation_name == 'app' or iZ_correlation_name == 'amm' or iZ_correlation_name == 'akk'):

                                    q1_bin_name = SOURCE_BIN_NAME_LIST[a]
                                    q2_bin_name = SOURCE_BIN_NAME_LIST[b]
                                    q3_bin_name = SOURCE_BIN_NAME_LIST[c]

                                    q1_z_array_los[0,:] = qs_z_array_los[a]
                                    q2_z_array_los[0,:] = qs_z_array_los[b]
                                    q3_z_array_los[0,:] = qs_z_array_los[c]

                                    if (iZ_correlation_name == 'app'):
                                        iB_1_l_z_grid = iB_app_l_z_grid

                                    elif (iZ_correlation_name == 'amm'):
                                        iB_1_l_z_grid = iB_amm_l_z_grid

                                    elif (iZ_correlation_name == 'akk'):
                                        iB_1_l_z_grid = iB_akk_l_z_grid

                                elif (iZ_correlation_name == 'att' or iZ_correlation_name == 'agk'):

                                    q1_bin_name = SOURCE_BIN_NAME_LIST[a]
                                    q2_bin_name = LENS_BIN_NAME
                                    q3_bin_name = SOURCE_BIN_NAME_LIST[c]

                                    q1_z_array_los[:3,:] = qs_z_array_los[a]

                                    q2_z_array_los[0,:] = ql_b1_z_array_los
                                    q2_z_array_los[1,:] = ql_b2_z_array_los
                                    q2_z_array_los[2,:] = ql_bs2_z_array_los

                                    q3_z_array_los[:3,:] = qs_z_array_los[c]

                                    if (iZ_correlation_name == 'att'):
                                        iB_1_l_z_grid = iB_att_b1_l_z_grid
                                        iB_2_l_z_grid = iB_att_b2_l_z_grid
                                        iB_3_l_z_grid = iB_att_bs2_l_z_grid

                                    elif (iZ_correlation_name == 'agk'):
                                        iB_1_l_z_grid = iB_agk_b1_l_z_grid
                                        iB_2_l_z_grid = iB_agk_b2_l_z_grid
                                        iB_3_l_z_grid = iB_agk_bs2_l_z_grid                

                                elif (iZ_correlation_name == 'agg'):

                                    q1_bin_name = SOURCE_BIN_NAME_LIST[a]
                                    q2_bin_name = LENS_BIN_NAME
                                    q3_bin_name = LENS_BIN_NAME

                                    q1_z_array_los[:4,:] = qs_z_array_los[a]

                                    q2_z_array_los[0,:] = ql_b1_z_array_los
                                    q2_z_array_los[1,:] = ql_b1_z_array_los
                                    q2_z_array_los[2,:] = ql_b1_z_array_los
                                    q2_z_array_los[3,:] = ql_b1_z_array_los

                                    q3_z_array_los[0,:] = ql_b1_z_array_los
                                    q3_z_array_los[1,:] = ql_b2_z_array_los
                                    q3_z_array_los[2,:] = ql_bs2_z_array_los
                                    q3_z_array_los[3,:] = ql_1_over_n_bar_z_array_los / 2.

                                    iB_1_l_z_grid = iB_agg_b1sq_l_z_grid
                                    iB_2_l_z_grid = iB_agg_b1_b2_l_z_grid
                                    iB_3_l_z_grid = iB_agg_b1_bs2_l_z_grid
                                    iB_4_l_z_grid = iB_agg_eps_epsdelta_l_z_grid

                                if (iZ_correlation_name == 'gpp' or iZ_correlation_name == 'gmm'  or iZ_correlation_name == 'gkk'):

                                    q1_bin_name = LENS_BIN_NAME
                                    q2_bin_name = SOURCE_BIN_NAME_LIST[b]
                                    q3_bin_name = SOURCE_BIN_NAME_LIST[c]

                                    q1_z_array_los[0,:] = ql_b1_z_array_los
                                    q1_z_array_los[1,:] = ql_b2_z_array_los
                                    q1_z_array_los[2,:] = ql_bs2_z_array_los

                                    q2_z_array_los[:3,:] = qs_z_array_los[b]

                                    q3_z_array_los[:3,:] = qs_z_array_los[c]

                                    if (iZ_correlation_name == 'gpp'):
                                        iB_1_l_z_grid = iB_gpp_b1_l_z_grid
                                        iB_2_l_z_grid = iB_gpp_b2_l_z_grid
                                        iB_3_l_z_grid = iB_gpp_bs2_l_z_grid

                                    elif (iZ_correlation_name == 'gmm'):
                                        iB_1_l_z_grid = iB_gmm_b1_l_z_grid
                                        iB_2_l_z_grid = iB_gmm_b2_l_z_grid
                                        iB_3_l_z_grid = iB_gmm_bs2_l_z_grid

                                    elif (iZ_correlation_name == 'gkk'):
                                        iB_1_l_z_grid = iB_gkk_b1_l_z_grid
                                        iB_2_l_z_grid = iB_gkk_b2_l_z_grid
                                        iB_3_l_z_grid = iB_gkk_bs2_l_z_grid

                                elif (iZ_correlation_name == 'gtt' or iZ_correlation_name == 'ggk'):

                                    q1_bin_name = LENS_BIN_NAME
                                    q2_bin_name = LENS_BIN_NAME
                                    q3_bin_name = SOURCE_BIN_NAME_LIST[c]

                                    q1_z_array_los[0,:] = ql_b1_z_array_los
                                    q1_z_array_los[1,:] = ql_b1_z_array_los
                                    q1_z_array_los[2,:] = ql_b1_z_array_los
                                    q1_z_array_los[3,:] = ql_b1_z_array_los

                                    q2_z_array_los[0,:] = ql_b1_z_array_los
                                    q2_z_array_los[1,:] = ql_b2_z_array_los
                                    q2_z_array_los[2,:] = ql_bs2_z_array_los
                                    q2_z_array_los[3,:] = ql_1_over_n_bar_z_array_los / 2.

                                    q3_z_array_los[:4,:] = qs_z_array_los[c]

                                    if (iZ_correlation_name == 'gtt'):
                                        iB_1_l_z_grid = iB_gtt_b1sq_l_z_grid
                                        iB_2_l_z_grid = iB_gtt_b1_b2_l_z_grid
                                        iB_3_l_z_grid = iB_gtt_b1_bs2_l_z_grid
                                        iB_4_l_z_grid = iB_gtt_eps_epsdelta_l_z_grid
                                    
                                    elif (iZ_correlation_name == 'ggk'):
                                        iB_1_l_z_grid = iB_ggk_b1sq_l_z_grid
                                        iB_2_l_z_grid = iB_ggk_b1_b2_l_z_grid
                                        iB_3_l_z_grid = iB_ggk_b1_bs2_l_z_grid
                                        iB_4_l_z_grid = iB_ggk_eps_epsdelta_l_z_grid

                                elif (iZ_correlation_name == 'ggg'):

                                    q1_bin_name = LENS_BIN_NAME
                                    q2_bin_name = LENS_BIN_NAME
                                    q3_bin_name = LENS_BIN_NAME

                                    q1_z_array_los[0,:] = ql_b1_z_array_los
                                    q1_z_array_los[1,:] = ql_b1_z_array_los
                                    q1_z_array_los[2,:] = ql_b1_z_array_los
                                    q1_z_array_los[3,:] = ql_b1_z_array_los
                                    q1_z_array_los[4,:] = ql_n_bar_z_array_los

                                    q2_z_array_los[0,:] = ql_b1_z_array_los
                                    q2_z_array_los[1,:] = ql_b1_z_array_los
                                    q2_z_array_los[2,:] = ql_b1_z_array_los
                                    q2_z_array_los[3,:] = ql_b1_z_array_los
                                    q2_z_array_los[4,:] = ql_1_over_n_bar_z_array_los

                                    q3_z_array_los[0,:] = ql_b1_z_array_los
                                    q3_z_array_los[1,:] = ql_b2_z_array_los
                                    q3_z_array_los[2,:] = ql_bs2_z_array_los
                                    q3_z_array_los[3,:] = ql_1_over_n_bar_z_array_los / 2.
                                    q3_z_array_los[4,:] = ql_1_over_n_bar_z_array_los

                                    iB_1_l_z_grid = iB_ggg_b1cu_l_z_grid
                                    iB_2_l_z_grid = iB_ggg_b1sq_b2_l_z_grid
                                    iB_3_l_z_grid = iB_ggg_b1sq_bs2_l_z_grid
                                    iB_4_l_z_grid = iB_ggg_b1_eps_epsdelta_l_z_grid
                                    iB_5_l_z_grid = iB_ggg_eps_eps_eps_l_z_grid

                                iB_z_weight_array_los = H_inv_z_array_los * chi_inv_z_array_los**4 * q1_z_array_los * q2_z_array_los * q3_z_array_los # Integrated bispectrum LOS weighting

                                iB_1_l_z_grid_los = np.zeros([l_array.size, z_array_los.size])
                                iB_2_l_z_grid_los = np.zeros([l_array.size, z_array_los.size])
                                iB_3_l_z_grid_los = np.zeros([l_array.size, z_array_los.size])
                                iB_4_l_z_grid_los = np.zeros([l_array.size, z_array_los.size])
                                iB_5_l_z_grid_los = np.zeros([l_array.size, z_array_los.size])

                                for i in range(l_array.size):
                                    if (iZ_correlation_name in ['app', 'amm', 'att', 'agg', 'gpp', 'gmm', 'gtt', 'ggg', 'akk', 'agk', 'gkk', 'ggk']):
                                        iB_1_l_z_func = interpolate.interp1d(z_array, iB_1_l_z_grid[i], fill_value=(0,0), bounds_error=False)
                                        iB_1_l_z_grid_los[i] = iB_1_l_z_func(z_array_los)

                                    if (iZ_correlation_name in ['att', 'agg', 'gpp', 'gmm', 'gtt', 'ggg', 'agk', 'gkk', 'ggk']):
                                        iB_2_l_z_func = interpolate.interp1d(z_array, iB_2_l_z_grid[i], fill_value=(0,0), bounds_error=False)
                                        iB_2_l_z_grid_los[i] = iB_2_l_z_func(z_array_los)

                                    if (iZ_correlation_name in ['att', 'agg', 'gpp', 'gmm', 'gtt', 'ggg', 'agk', 'gkk', 'ggk']):
                                        iB_3_l_z_func = interpolate.interp1d(z_array, iB_3_l_z_grid[i], fill_value=(0,0), bounds_error=False)
                                        iB_3_l_z_grid_los[i] = iB_3_l_z_func(z_array_los)

                                    if (iZ_correlation_name in ['agg', 'gtt', 'ggg', 'ggk']):
                                        iB_4_l_z_func = interpolate.interp1d(z_array, iB_4_l_z_grid[i], fill_value=(0,0), bounds_error=False)
                                        iB_4_l_z_grid_los[i] = iB_4_l_z_func(z_array_los)

                                    if (iZ_correlation_name in ['ggg']):
                                        iB_5_l_z_func = interpolate.interp1d(z_array, iB_5_l_z_grid[i], fill_value=(0,0), bounds_error=False)
                                        iB_5_l_z_grid_los[i] = iB_5_l_z_func(z_array_los)

                                iB_l = np.zeros([5, l_array.size])

                                iB_1_l_z_integrand_grid = iB_1_l_z_grid_los * iB_z_weight_array_los[0,:]
                                iB_l[0] = np.trapezoid(iB_1_l_z_integrand_grid, x=z_array_los, axis=1)

                                iB_2_l_z_integrand_grid = iB_2_l_z_grid_los * iB_z_weight_array_los[1,:]
                                iB_l[1] = np.trapezoid(iB_2_l_z_integrand_grid, x=z_array_los, axis=1)

                                iB_3_l_z_integrand_grid = iB_3_l_z_grid_los * iB_z_weight_array_los[2,:]
                                iB_l[2] = np.trapezoid(iB_3_l_z_integrand_grid, x=z_array_los, axis=1)

                                iB_4_l_z_integrand_grid = iB_4_l_z_grid_los * iB_z_weight_array_los[3,:]
                                iB_l[3] = np.trapezoid(iB_4_l_z_integrand_grid, x=z_array_los, axis=1)

                                iB_5_l_z_integrand_grid = iB_5_l_z_grid_los * iB_z_weight_array_los[4,:]
                                iB_l[4] = np.trapezoid(iB_5_l_z_integrand_grid, x=z_array_los, axis=1)

                                np.savetxt(iB_spectra_path+'iB_'+iZ_correlation_name+'_l_'+q1_bin_name+'_'+q2_bin_name+'_'+q3_bin_name+filename_extension, iB_l.T)

                                ell, iB_1_ell = C_ell_spherical_sky(l_array, iB_l[0])
                                ell, iB_2_ell = C_ell_spherical_sky(l_array, iB_l[1])
                                ell, iB_3_ell = C_ell_spherical_sky(l_array, iB_l[2])
                                ell, iB_4_ell = C_ell_spherical_sky(l_array, iB_l[3])
                                ell, iB_5_ell = C_ell_spherical_sky(l_array, iB_l[4])

                                #if nside is given, consider the pixel window function's effect on the C_ells
                                if nside is not None and healpy_installed:
                                    
                                    pixwin = hp.pixwin(nside, lmax=np.max(ell))
                                    pixwin_ell = np.arange(len(pixwin))                                         # pixwin creates window function for ell=0 to 3*nside-1
                                    pixwin = pixwin[np.intersect1d(ell, pixwin_ell, return_indices=True)[2]]    # match pixwin function to correct ells
                                    pixwin = np.append(pixwin, np.zeros(len(ell) - len(pixwin)))                # Add zeros to match length
                                    
                                    # Add correction to C_ell
                                    iB_1_ell = iB_1_ell * pixwin**2             
                                    iB_2_ell = iB_2_ell * pixwin**2
                                    iB_3_ell = iB_3_ell * pixwin**2
                                    iB_4_ell = iB_4_ell * pixwin**2
                                    iB_5_ell = iB_5_ell * pixwin**2
                                    

                                iB_4_ell = iB_4_ell - np.mean(iB_4_ell[ell > 1000]) # for the shot noise terms
                                iB_5_ell = iB_5_ell - np.mean(iB_5_ell[ell > 1000]) # for the shot noise terms

                                #iZ_array_bin_averaged = np.zeros([5, alpha_min_arcmins.size])
                                iZ_array_bin_averaged = np.zeros([6, alpha_min_arcmins.size]) ## for testing if the sum of iB_ells before Harmonic transform helps or not in any way

                                if (iZ_correlation_name == 'app'):
                                    iZ_array_bin_averaged[0] = iZp_theta_bin_averaged(ell, iB_1_ell)
                                    iZ_array_bin_averaged[5] = iZ_array_bin_averaged[0] ##
                                elif (iZ_correlation_name == 'amm'):
                                    iZ_array_bin_averaged[0] = iZm_theta_bin_averaged(ell, iB_1_ell)
                                    iZ_array_bin_averaged[5] = iZ_array_bin_averaged[0] ##
                                elif (iZ_correlation_name == 'att'):
                                    iZ_array_bin_averaged[0] = iZt_theta_bin_averaged(ell, iB_1_ell)
                                    iZ_array_bin_averaged[1] = iZt_theta_bin_averaged(ell, iB_2_ell)
                                    iZ_array_bin_averaged[2] = iZt_theta_bin_averaged(ell, iB_3_ell)
                                    iZ_array_bin_averaged[5] = iZt_theta_bin_averaged(ell, iB_1_ell+iB_2_ell+iB_3_ell) ##
                                elif (iZ_correlation_name == 'agg'):
                                    iZ_array_bin_averaged[0] = iZ_theta_bin_averaged(ell, iB_1_ell)
                                    iZ_array_bin_averaged[1] = iZ_theta_bin_averaged(ell, iB_2_ell)
                                    iZ_array_bin_averaged[2] = iZ_theta_bin_averaged(ell, iB_3_ell)    
                                    iZ_array_bin_averaged[3] = iZ_theta_bin_averaged(ell, iB_4_ell)   
                                    iZ_array_bin_averaged[5] = iZ_theta_bin_averaged(ell, iB_1_ell+iB_2_ell+iB_3_ell+iB_4_ell) ##
                                elif (iZ_correlation_name == 'gpp'):
                                    iZ_array_bin_averaged[0] = iZp_theta_bin_averaged(ell, iB_1_ell)
                                    iZ_array_bin_averaged[1] = iZp_theta_bin_averaged(ell, iB_2_ell)
                                    iZ_array_bin_averaged[2] = iZp_theta_bin_averaged(ell, iB_3_ell) 
                                    iZ_array_bin_averaged[5] = iZp_theta_bin_averaged(ell, iB_1_ell+iB_2_ell+iB_3_ell) ##
                                elif (iZ_correlation_name == 'gmm'):
                                    iZ_array_bin_averaged[0] = iZm_theta_bin_averaged(ell, iB_1_ell)
                                    iZ_array_bin_averaged[1] = iZm_theta_bin_averaged(ell, iB_2_ell)
                                    iZ_array_bin_averaged[2] = iZm_theta_bin_averaged(ell, iB_3_ell) 
                                    iZ_array_bin_averaged[5] = iZm_theta_bin_averaged(ell, iB_1_ell+iB_2_ell+iB_3_ell) ##
                                elif (iZ_correlation_name == 'gtt'):
                                    iZ_array_bin_averaged[0] = iZt_theta_bin_averaged(ell, iB_1_ell)
                                    iZ_array_bin_averaged[1] = iZt_theta_bin_averaged(ell, iB_2_ell)
                                    iZ_array_bin_averaged[2] = iZt_theta_bin_averaged(ell, iB_3_ell)    
                                    iZ_array_bin_averaged[3] = iZt_theta_bin_averaged(ell, iB_4_ell)   
                                    iZ_array_bin_averaged[5] = iZt_theta_bin_averaged(ell, iB_1_ell+iB_2_ell+iB_3_ell+iB_4_ell) ##    
                                elif (iZ_correlation_name == 'ggg' or iZ_correlation_name == 'akk' or iZ_correlation_name == 'agk' or iZ_correlation_name == 'gkk' or iZ_correlation_name == 'ggk'):
                                    iZ_array_bin_averaged[0] = iZ_theta_bin_averaged(ell, iB_1_ell)
                                    iZ_array_bin_averaged[1] = iZ_theta_bin_averaged(ell, iB_2_ell)
                                    iZ_array_bin_averaged[2] = iZ_theta_bin_averaged(ell, iB_3_ell)    
                                    iZ_array_bin_averaged[3] = iZ_theta_bin_averaged(ell, iB_4_ell)   
                                    iZ_array_bin_averaged[4] = iZ_theta_bin_averaged(ell, iB_5_ell) 
                                    iZ_array_bin_averaged[5] = iZ_theta_bin_averaged(ell, iB_1_ell+iB_2_ell+iB_3_ell+iB_4_ell+iB_5_ell) ##    

                                np.savetxt(iZ_correlations_path+'iZ_'+iZ_correlation_name+'_'+q1_bin_name+'_'+q2_bin_name+'_'+q3_bin_name+filename_extension, iZ_array_bin_averaged.T)  

                            corr_idx += 1 

            end_spectra_and_correlations = time.time()
            print('Computing P(l) and/or iB(l) spectra and xi(alpha) and/or iZ(alpha) correlations took %ss'%(end_spectra_and_correlations - start_spectra_and_correlations), flush=True) 

        #################################################################################################################################
        #################################################################################################################################
        # STEP 4: Delete classy.CLASS object and helper cosmology object
        #################################################################################################################################
        #################################################################################################################################

        del CosmoClassObject
        cclass.struct_cleanup()  
        print('\nDeleted CosmoClassObject and cleaned up classy.CLASS object', flush=True)
        
        #################################################################################################################################
        #################################################################################################################################

    if (pool_opened == True):
        pool.close()
        pool.join()
        pool_opened = False
        print('\nPool closed', flush=True)

    end_program = time.time()
    print('\nTime taken for execution of the whole script (seconds):', end_program - start_program, flush=True) 

#####################

# Run the main_function

if platform.system() == "Darwin":  # macOS
    if __name__ == '__main__':
        print('Running on macOS system', flush=True)
        main_function()
    
else: 
    print('Running on non-macOS system', flush=True)
    main_function()