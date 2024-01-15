import numpy as np

### (l,z) grid settings
l_array = np.unique(np.logspace(np.log10(2), np.log10(15000), 113).astype(int)) # 100 integer l values
#z_array = np.append(np.array([0.01, 0.03, 0.07]), np.logspace(np.log10(0.1), np.log10(2.0), 27)) # 30 z values
z_array = np.append(np.array([0.01, 0.03, 0.07]), np.logspace(np.log10(0.1), np.log10(3.6), 30)) # 30 z values

### This is a config file where we give what the input parameters for the main.py file are

### T17 fiducial cosmology
#cosmo_pars_fid = {'Omega_b': 0.046, 'Omega_m': 0.279, 'h': 0.7, 'n_s': 0.97, 'sigma8': 0.82, 'w_0': -1.0, 'w_a': 0.0, 'c_min': 3.13, 'eta_0': 0.603, 'Mv': 0.0} 
#cosmo_pars_fid = {'Omega_b': 0.046, 'Omega_m': 0.279, 'h': 0.7, 'n_s': 0.97, 'A_s': np.exp(np.log(10**10 * 2.19685e-9))/(10**10), 'w_0': -1.0, 'w_a': 0.0, 'c_min': 3.13, 'eta_0': 0.603, 'Mv': 0.0} 

### COSMOGRIDV1 fiducial cosmology
cosmo_pars_fid = {'Omega_b': 0.0493, 'Omega_m': 0.26, 'h': 0.673, 'n_s': 0.9649, 'sigma8': 0.84, 'w_0': -1.0, 'w_a': 0.0, 'c_min': 3.13, 'eta_0': 0.603, 'Mv': 0.0}

#params_lhs = np.load('../data/cosmo_parameters/test_iB_LHS_5parameter_file_5e2points.npz')
params_lhs = {}

########################################################
########################################################

P3D_set_k_gt_k_nl_to_zero = False

P3D_type = 'nl' # lin, nl
compute_P_grid = 'yes' # yes, no
compute_P_spectra_and_correlations = 'yes' # yes, no
B3D_type = 'nl' # lin, nl
compute_iB_grid = 'no' # yes, no
compute_iB_spectra_and_correlations = 'no' # yes, no
compute_area_prefactor = 'no' # yes, no

compute_chi_D_values = 'no'
compute_H_values = 'no'
use_Dirac_comb = True

# can also append other distinguishing suffixes e.g. 'shear_x_shear_SkySim5000'
grid_type = 'shear_x_shear_COSMOGRIDV1'
spectra_and_correlation_type = 'shear_x_shear_DESY3_COSMOGRIDV1_Dirac_center'

### correlation name list
# Naming convention aperture : a : shear aperture mass ; g : galaxy mean number density in tophat filter
# Naming convention 2PCF : pp : xi_plus ; mm : xi_minus ; tt : xi_tangential_shear ; gg : xi_galaxy_clustering

# choose one of the following (comment the ones completely which are not being used!)
# shear x shear (when computing correlations with shear only)
xi_correlation_name_list = ['pp', 'mm']
iZ_correlation_name_list = ['app', 'amm']

# halo/galaxy x kappa (when computing correlations with tracer x kappa)
#xi_correlation_name_list = ['kk', 'gk', 'gg']
#iZ_correlation_name_list = ['akk', 'agk', 'agg', 'gkk', 'ggk', 'ggg']

# halo/galaxy x shear (when computing correlations with tracer x shear)
#xi_correlation_name_list = ['pp', 'mm', 'tt', 'gg']
#iZ_correlation_name_list = ['app', 'amm', 'att', 'agg', 'gpp', 'gmm', 'gtt', 'ggg']

### radius of the compensated and tophat filters
theta_U_arcmins = 90
theta_T_arcmins = 90

### Minimum, maximum separations for the correlation computations and the number of log bins

# for global 2PCF
min_sep_tc_xi = 5
max_sep_tc_xi = 250
nbins_tc_xi = 10

# for local (within filter) 2PCF
min_sep_tc = 5
max_sep_tc = 2*theta_T_arcmins - 10
if (theta_T_arcmins == 30):
    nbins_tc = 5
elif (theta_T_arcmins == 50):
    nbins_tc = 7
elif (theta_T_arcmins == 90):
    nbins_tc = 10
elif (theta_T_arcmins == 130):
    nbins_tc = 15

### SOURCE bin name and values
#SOURCE_BIN_NAME_LIST = ['BIN12', 'BIN34']
#SOURCE_BIN_VALUES = ['', '']

#SOURCE_BIN_NAME_LIST = ['zs1', 'zs2']
#SOURCE_BIN_VALUES = [0.5739, 1.0334]

#SOURCE_BIN_NAME_LIST = ['SBIN4']
#SOURCE_BIN_VALUES = ['']

SOURCE_BIN_NAME_LIST = ['SBIN1', 'SBIN2', 'SBIN3', 'SBIN4']
SOURCE_BIN_VALUES = ['', '', '', '']

SOURCE_BIN_delta_photoz_values = [0.0]
SOURCE_BIN_m_values = [1]
A_IA_0_NLA = 0.0
alpha_IA_0_NLA = 0.0

### LENS bin name and type (only used when computing correlations with halos x lensing)
halo_type_user = '' # 'all_halos'

#LENS_BIN_FULLNAME_theory = '_redmagicLBIN1like'
#LENS_BIN_FULLNAME_theory = '_maglimLBIN1like'
LENS_BIN_FULLNAME_theory = '_maglimLBIN3like'

if ('Lens1like' in LENS_BIN_FULLNAME_theory):
    LENS_BIN_NAME = 'LBIN1'
elif ('Lens3like' in LENS_BIN_FULLNAME_theory):
    LENS_BIN_NAME = 'LBIN3'

if (use_Dirac_comb == True):
    LENS_BIN_FULLNAME_theory = LENS_BIN_FULLNAME_theory + '_source_BINS_weights_Dirac_comb/'
else:
    LENS_BIN_FULLNAME_theory = LENS_BIN_FULLNAME_theory + '/'

### Choose iB_app, iB_amm nl types
if (B3D_type == 'nl'):
    iB_app_type = 'GM' 
    iB_amm_type = 'GMRF'
    iB_att_type = 'GMRF'
    iB_agg_type = 'GM'

    iB_gpp_type = 'GM'
    iB_gmm_type = 'GMRF'
    iB_gtt_type = 'GMRF'
    iB_ggg_type = 'GM'

f_sq = 7

### paths to save/load grids, spectra and correlations
P_l_z_grid_path = "../output/grids/P_"+P3D_type+"_grids_l"+str(l_array.size)+"_z"+str(z_array.size)+"_"+grid_type+"/"

if (B3D_type == 'lin'):
    iB_l_z_grid_path = "../output/grids/iB_U"+str(theta_U_arcmins)+"W"+str(theta_T_arcmins)+"W"+str(theta_T_arcmins)+"_"+B3D_type+"_grids_l"+str(l_array.size)+"_z"+str(z_array.size)+"_"+grid_type+"/"
elif (B3D_type == 'nl'):
    iB_l_z_grid_path = "../output/grids/iB_U"+str(theta_U_arcmins)+"W"+str(theta_T_arcmins)+"W"+str(theta_T_arcmins)+"_"+B3D_type+"_grids_l"+str(l_array.size)+"_z"+str(z_array.size)+"_J0_GM_J2J4_GMRF_fsq7_"+grid_type+"/"

if (compute_P_spectra_and_correlations == 'yes'):
    P_spectra_path = "../output/spectra/P_"+P3D_type+"_spectra_l"+str(l_array.size)+"_"+spectra_and_correlation_type+"/"
    xi_correlations_path = "../output/correlations/xi_"+P3D_type+"_correlations_alpha_"+str(min_sep_tc_xi)+"_"+str(max_sep_tc_xi)+"_"+str(nbins_tc_xi)+"_"+spectra_and_correlation_type+"/"

if (compute_iB_spectra_and_correlations == 'yes'):
    if (B3D_type == 'lin'):
        iB_spectra_path = "../output/spectra/iB_U"+str(theta_U_arcmins)+"W"+str(theta_T_arcmins)+"W"+str(theta_T_arcmins)+"_"+B3D_type+"_spectra_l"+str(l_array.size)+"_"+spectra_and_correlation_type+"/"
        iZ_correlations_path = "../output/correlations/iZ_U"+str(theta_U_arcmins)+"W"+str(theta_T_arcmins)+"W"+str(theta_T_arcmins)+"_"+B3D_type+"_correlations_alpha_"+str(min_sep_tc)+"_"+str(max_sep_tc)+"_"+str(nbins_tc)+"_"+spectra_and_correlation_type+"/"
    
    elif (B3D_type == 'nl'):
        iB_spectra_path = "../output/spectra/iB_U"+str(theta_U_arcmins)+"W"+str(theta_T_arcmins)+"W"+str(theta_T_arcmins)+"_"+B3D_type+"_spectra_l"+str(l_array.size)+"_J0_GM_J2J4_GMRF_fsq7_"+spectra_and_correlation_type+"/"
        iZ_correlations_path = "../output/correlations/iZ_U"+str(theta_U_arcmins)+"W"+str(theta_T_arcmins)+"W"+str(theta_T_arcmins)+"_"+B3D_type+"_correlations_alpha_"+str(min_sep_tc)+"_"+str(max_sep_tc)+"_"+str(nbins_tc)+"_J0_GM_J2J4_GMRF_fsq7_"+spectra_and_correlation_type+"/"

if (compute_area_prefactor == 'yes'):
    area_prefactor_path = "../output/area_prefactor/"