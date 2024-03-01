import numpy as np

### (l,z) grid settings
#l_array = np.unique(np.logspace(np.log10(2), np.log10(15000), 113).astype(int)) # 100 integer l values
#l_array = np.unique(np.logspace(np.log10(2), np.log10(15000), 247).astype(int)) # 200 integer l values
l_array = np.unique(np.logspace(np.log10(2), np.log10(15000), 719).astype(int)) # 500 integer l values

#z_array = np.append(np.array([0.01, 0.03, 0.07]), np.logspace(np.log10(0.1), np.log10(2.0), 27)) # 30 z values
#z_array = np.append(np.array([0.01, 0.03, 0.07]), np.logspace(np.log10(0.1), np.log10(3.5), 35)) # 30 z values # for Dirac COSMOGRIDV1

#The following z_array are the mean redshifts of the COSMOGRIDV1 fiducial particle shells.
z_array = np.array([0.006387739907950163,
                    0.0192780252546072,0.03240172937512398,0.04576645791530609,0.05938002094626427,0.0732506513595581,
                    0.08738701045513153,0.10179822146892548,0.11649389564990997,0.13148410618305206,0.14677950739860535,
                    0.16239139437675476,0.1783316433429718,0.1946128010749817,0.21124814450740814,0.22825175523757935,
                    0.24563850462436676,0.26342421770095825,0.2816255986690521,0.3002605438232422,0.31934794783592224,
                    0.33890801668167114,0.35896217823028564,0.37953343987464905,0.40064623951911926,0.42232686281204224,
                    0.44460320472717285,0.4675053358078003,0.4910655915737152,0.5153185129165649,0.5403013229370117,
                    0.5660544037818909,0.5926209688186646,0.6200478076934814,0.6483858823776245,0.6776900291442871,
                    0.7080200910568237,0.7394411563873291,0.7720241546630859,0.8058466911315918,0.8409937024116516,
                    0.8775583505630493,0.9156432151794434,0.955361545085907,0.9968384504318237,1.0402125120162964,
                    1.0856380462646484,1.1332874298095703,1.1833534240722656,1.2360525131225586,1.2916284799575806,
                    1.350358486175537,1.4125570058822632,1.478582501411438,1.5488479137420654,1.6238294839859009,
                    1.704079508781433,1.7902441024780273,1.8830831050872803,1.983497142791748,2.0925631523132324,
                    2.2115797996520996,2.342129945755005,2.48616361618042,2.646116018295288,2.825070858001709,
                    3.026996612548828,3.2570996284484863,3.4399685859680176])

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
compute_iB_grid = 'yes' # yes, no
compute_iB_spectra_and_correlations = 'yes' # yes, no
compute_area_prefactor = 'yes' # yes, no

compute_chi_D_values = 'no'
compute_H_values = 'no'
use_Dirac_comb = False

nside = 512 # Set to None to ignore

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
min_sep_tc_xi = 10
max_sep_tc_xi = 250
nbins_tc_xi = 15

# TODO: generalize
# for local (within filter) 2PCF
min_sep_tc = 10
max_sep_tc = 250
nbins_max = 15

binedges = np.geomspace(theta_T_arcmins, max_sep_tc, nbins_max+1)
binedges = binedges[np.where(binedges <= 2 * theta_T_arcmins - 5)]
nbins_tc = len(binedges)-1
max_sep_tc = binedges[-1]


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
