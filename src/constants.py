import numpy as np
import multiprocessing as mp

### k_min and k_max [1/Mpc]; z_min and z_max settings for power spectrum of the CosmoClass grid
_k_min_ = 5.e-4 # [1/Mpc]
_k_max_ = 150.0 # [1/Mpc]

_z_min_ = 0.0
_z_max_ = 2.5 # note that the G_K response function can only be computed up to a max z = 2.63 and G_1 up to z = 3.0
#_z_max_ = 3.5 # for COSMOGRIDV1

### k_max_pk [1/Mpc] and z_max_pk settings for CLASS power spectrum (set these to larger values than required for _k_max_ and _z_max_)
#_k_max_pk_ = 70.0 # for tracer x lensing
_k_max_pk_ = 1.2*_k_max_
_z_max_pk_ = 1.2*_z_max_ 

_k_grid_npts_ = 500
_z_grid_zstep_ = 0.05

### R_min and R_max [Mpc] settings for sigma(R) quantities in CosmoClass

_compute_R_z_grid_ = False

_R_min_ = 0.1 # [Mpc]
_R_max_ = 50.0 # [Mpc]

### GMRF _f_sq_ value
_f_sq_ = 7

### f_NL value
_f_NL_ = 0.0
_f_NL_type_ = 'local' # 'local' or 'equilateral' or 'orthogonal'

### l_max for 2D spectra (set it to an integer value)
_l_max_ = 15000

### number of redshift bins used in LOS projection (linear spacing)
_N_los_ = int(_z_max_*100)

### for halo quantities
_M_SUN_ = 1.98847e30 # Solar mass in kg
_Mpc_over_m_ = 3.085677581282e22  # conversion factor from meters to megaparsecs
_Gyr_over_Mpc_ = 3.06601394e2 # conversion factor from megaparsecs to gigayears (c=1 units, Julian years of 365.25 days)
_c_ = 2.99792458e8 # c in m/s
_G_ = 6.67428e-11 # Newton constant in m^3/Kg/s^2
_rho_class_to_SI_units_ = 3.*_c_*_c_/8./np.pi/_G_/_Mpc_over_m_/_Mpc_over_m_
_kg_m3_to_M_sun_Mpc3_units_ = 1./_M_SUN_*_Mpc_over_m_*_Mpc_over_m_*_Mpc_over_m_
_delta_c_ = 1.686
_Delta_ = 200.0

### DES like f_sky
_f_sky_ = 5000/41252.96125

### number of parallel processors for pool
_num_of_prcoesses_ = mp.cpu_count()-1

# Takahashi et al. (2017) Appendix B eqn (28) fitting formula for finite shell thickness - parameter values
_apply_T17_corrections_ = False
_c1_T17_ = 9.5171e-4
_c2_T17_ = 5.1543e-3
_a1_T17_ = 1.3063
_a2_T17_ = 1.1475
_a3_T17_ = 0.62793
#_ell_res_T17_ = 1.6*4096
_ell_res_T17_ = 1.6*2048
_box_sizes_T17_ = np.arange(1,15)*450.0 # [Mpc/h]
_k_h_over_Mpc_box_sizes_T17_ = 2*np.pi/_box_sizes_T17_