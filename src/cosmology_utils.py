import numpy as np
#from misc_utils import function_interp2d_grid, function_interp2d_RectBivariateSpline_grid, function_interp2d_RectBivariateSpline_grid_parallel
from scipy import interpolate
from bispectrum import *
from scipy.optimize import fsolve
from scipy.signal import find_peaks
#import itertools
#import multiprocessing as mp
import constants

class CosmoClass:
    def __init__(self, cclass, k_min_max=[constants._k_min_,constants._k_max_], z_min_max=[constants._z_min_,constants._z_max_]):
     
        ## store parameters
        self.sigma8_0 = cclass.sigma8()
        self.n_s = cclass.n_s()
        self.h = cclass.h()
        self.Omega0_m = cclass.Omega0_m()
        self.f_sq = constants._f_sq_
        self.f_NL = constants._f_NL_
        self.f_NL_type = constants._f_NL_type_

        self.k_min = k_min_max[0]
        self.k_max = k_min_max[1]
        self.z_min = z_min_max[0]
        self.z_max = z_min_max[1]

        ### tabulate background quantities
        self.chi_z = interpolate.interp1d(cclass.get_background()['z'], cclass.get_background()['comov. dist.'], kind='cubic', fill_value=0.0)
        self.z_chi = interpolate.interp1d(cclass.get_background()['comov. dist.'], cclass.get_background()['z'], kind='cubic', fill_value=0.0)
        self.H_z = interpolate.interp1d(cclass.get_background()['z'], cclass.get_background()['H [1/Mpc]'], kind='cubic', fill_value=0.0)
        self.D_plus_z = interpolate.interp1d(cclass.get_background()['z'], cclass.get_background()['gr.fac. D'], kind='cubic', fill_value=0.0) # normalised to 1 at z=0
        self.T_k = interpolate.interp1d(cclass.get_transfer()['k (h/Mpc)']*self.h, cclass.get_transfer()['d_tot']/cclass.get_transfer()['d_tot'][0], kind='cubic', fill_value=0.0) # normalised to 1 on large scales
        
        self.rho_crit_z = interpolate.interp1d(cclass.get_background()['z'], cclass.get_background()['(.)rho_crit']*constants._rho_class_to_SI_units_* constants._kg_m3_to_M_sun_Mpc3_units_ , kind='cubic', fill_value=0.0) # in [M_sun/Mpc^3]
        self.rho0_m = self.Omega0_m * self.rho_crit_z(0.) # in [M_sun/Mpc^3]

        ## k,z grid points
        #ClassyClassObject_k_grid_points = cclass.get_pk_and_k_and_z()[1]
        #self.k_grid_points_ascending = ClassyClassObject_k_grid_points[(ClassyClassObject_k_grid_points >= self.k_min ) & (ClassyClassObject_k_grid_points <= self.k_max)]

        #ClassyClassObject_z_grid_points = cclass.get_pk_and_k_and_z()[2]
        #self.z_grid_points_decending = ClassyClassObject_z_grid_points[(ClassyClassObject_z_grid_points >= self.z_min ) & (ClassyClassObject_z_grid_points <= self.z_max)]
        #self.z_grid_points_ascending = np.flip(self.z_grid_points_decending)

        self.k_grid_points_ascending = np.logspace(np.log10(self.k_min), np.log10(self.k_max), constants._k_grid_npts_)
        self.z_grid_points_ascending = np.arange(self.z_min, self.z_max + constants._z_grid_zstep_, step = constants._z_grid_zstep_)

        ### tabulate power spectrum (with RectBivariateSpline interpolation)
        #self.P3D_k_z_lin = function_interp2d_RectBivariateSpline_grid(self.k_grid_points_ascending, self.z_grid_points_ascending, cclass.pk_lin)
        #self.P3D_k_z_nl = function_interp2d_RectBivariateSpline_grid(self.k_grid_points_ascending, self.z_grid_points_ascending, cclass.pk)

        ## new way
        P3D_k_z_lin_grid_points = cclass.get_pk_array(self.k_grid_points_ascending, self.z_grid_points_ascending, self.k_grid_points_ascending.size, self.z_grid_points_ascending.size, 0).reshape(self.z_grid_points_ascending.size, self.k_grid_points_ascending.size).T
        P3D_k_z_nl_grid_points = cclass.get_pk_array(self.k_grid_points_ascending, self.z_grid_points_ascending, self.k_grid_points_ascending.size, self.z_grid_points_ascending.size, 1).reshape(self.z_grid_points_ascending.size, self.k_grid_points_ascending.size).T

        self.P3D_k_z_lin = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, P3D_k_z_lin_grid_points)
        self.P3D_k_z_nl = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, P3D_k_z_nl_grid_points)

        ### tabulate linear power spectrum tilt

        '''
        # with cclass.pk_tilt function
        n_eff_lin_k_grid_points = np.zeros(self.k_grid_points_ascending.size)
        for i in range(self.k_grid_points_ascending.size):
            n_eff_lin_k_grid_points[i] = cclass.pk_tilt(self.k_grid_points_ascending[i], 0.0) # pk_tilt of class gives the logarithmic derivative wrt linear Pk
        
        self.n_eff_lin_k = interpolate.interp1d(self.k_grid_points_ascending, n_eff_lin_k_grid_points, kind='cubic', fill_value=0.0)
        self.n_eff_lin_k_smoothed = interpolate.UnivariateSpline(self.k_grid_points_ascending, n_eff_lin_k_grid_points)
        self.n_eff_lin_k_smoothed.set_smoothing_factor(1.8)
        '''

        # with numpy gradient
        n_eff_lin_k_grid_points = np.gradient(np.log(P3D_k_z_lin_grid_points[:, 0]), np.log(self.k_grid_points_ascending)) # can be computed at any redshift (as this is for linear Pk)
        self.n_eff_lin_k = interpolate.interp1d(self.k_grid_points_ascending, n_eff_lin_k_grid_points, kind='cubic', fill_value=0.0)

        k_BAO_min = 0.01
        k_BAO_max = 1.0
        k_BAO_mask = (self.k_grid_points_ascending > k_BAO_min) & (self.k_grid_points_ascending < k_BAO_max)

        # Identify maxima of the BAO region
        max_BAO_peaks, _ = find_peaks( n_eff_lin_k_grid_points[k_BAO_mask])

        # Identify minima by inverting the data
        min_BAO_peaks, _ = find_peaks(-n_eff_lin_k_grid_points[k_BAO_mask])

        if len(max_BAO_peaks) > len(min_BAO_peaks):
            max_BAO_peaks = max_BAO_peaks[:len(min_BAO_peaks)]
        elif len(min_BAO_peaks) > len(max_BAO_peaks):
            min_BAO_peaks = min_BAO_peaks[:len(max_BAO_peaks)]

        k_grid_points_ascending_BAO = (self.k_grid_points_ascending[k_BAO_mask][min_BAO_peaks] + self.k_grid_points_ascending[k_BAO_mask][max_BAO_peaks])/2
        n_eff_lin_k_grid_points_BAO = (n_eff_lin_k_grid_points[k_BAO_mask][min_BAO_peaks] + n_eff_lin_k_grid_points[k_BAO_mask][max_BAO_peaks])/2

        k_grid_points_ascending_BAO_smoothed = np.concatenate((self.k_grid_points_ascending[self.k_grid_points_ascending <= k_BAO_min], k_grid_points_ascending_BAO, self.k_grid_points_ascending[self.k_grid_points_ascending >= k_BAO_max]))
        n_eff_lin_k_grid_points_BAO_smoothed = np.concatenate((n_eff_lin_k_grid_points[self.k_grid_points_ascending <= k_BAO_min], n_eff_lin_k_grid_points_BAO, n_eff_lin_k_grid_points[self.k_grid_points_ascending >= k_BAO_max]))

        self.n_eff_lin_k_smoothed = interpolate.UnivariateSpline(k_grid_points_ascending_BAO_smoothed, n_eff_lin_k_grid_points_BAO_smoothed)
        self.n_eff_lin_k_smoothed.set_smoothing_factor(0.002)

        ### tabulate non-linear scale from linear power spectrum 
        k_nl_z_grid_points = np.zeros(self.z_grid_points_ascending.size)
        for j in range(self.z_grid_points_ascending.size):
            k_nl_z_grid_points[j] = self.compute_k_nl_z_pk_lin(self.z_grid_points_ascending[j])

        self.k_nl_z_pk_lin = interpolate.interp1d(self.z_grid_points_ascending, k_nl_z_grid_points, kind='cubic', fill_value=0.0)

        ### tabulate quantities for various biscpectrum recipes (e.g. GM, response functions etc, nonlinear Pk tilt etc.) in k,z grid (parallel)
        self.RF_G_1_table_data_loaded = False
        self.RF_G_K_table_data_loaded = False
   
        self.compute_RF_G_1_k_z(0.5, 0.2) # just an initial call with random arguments to simply load the G1 and GK tables
        self.compute_RF_G_K_k_z(0.5, 0.2)

        #'''
        ############################################################################################################

        # compute grid quantities serially

        B3D_a_GM_k_z_grid_points = np.zeros((self.k_grid_points_ascending.size, self.z_grid_points_ascending.size))
        B3D_b_GM_k_z_grid_points = np.zeros((self.k_grid_points_ascending.size, self.z_grid_points_ascending.size))
        B3D_c_GM_k_z_grid_points = np.zeros((self.k_grid_points_ascending.size, self.z_grid_points_ascending.size))
        RF_G_1_k_z_grid_points   = np.zeros((self.k_grid_points_ascending.size, self.z_grid_points_ascending.size))
        RF_G_K_k_z_grid_points   = np.zeros((self.k_grid_points_ascending.size, self.z_grid_points_ascending.size))
        n_eff_nl_k_z_grid_points = np.zeros((self.k_grid_points_ascending.size, self.z_grid_points_ascending.size))
        M_k_z_grid_points = np.zeros((self.k_grid_points_ascending.size, self.z_grid_points_ascending.size)) # poisson factor

        for j in range(self.z_grid_points_ascending.size):

            z = self.z_grid_points_ascending[j]
            sigma8z = self.sigma8_z(z)

            n_eff_nl_k_z_grid_points[:,j] = np.gradient(np.log(P3D_k_z_nl_grid_points[:, j]), np.log(self.k_grid_points_ascending))

            for i in range(self.k_grid_points_ascending.size):

                k = self.k_grid_points_ascending[i]
                n_eff = self.n_eff_lin_k_smoothed(k)
                q = k/k_nl_z_grid_points[j]  # non-linearity ratio

                B3D_a_GM_k_z_grid_points[i][j] = a_GM(n_eff, q, sigma8z)
                B3D_b_GM_k_z_grid_points[i][j] = b_GM(n_eff, q)
                B3D_c_GM_k_z_grid_points[i][j] = c_GM(n_eff, q)
                RF_G_1_k_z_grid_points[i][j] = self.compute_RF_G_1_k_z(k, z)
                RF_G_K_k_z_grid_points[i][j] = self.compute_RF_G_K_k_z(k, z)
                M_k_z_grid_points[i][j] = self.compute_M_k_z(k, z)

        self.B3D_a_GM_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, B3D_a_GM_k_z_grid_points)
        self.B3D_b_GM_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, B3D_b_GM_k_z_grid_points)
        self.B3D_c_GM_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, B3D_c_GM_k_z_grid_points)
        self.RF_G_1_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, RF_G_1_k_z_grid_points)
        self.RF_G_K_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, RF_G_K_k_z_grid_points)
        self.n_eff_nl_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, n_eff_nl_k_z_grid_points)
        self.M_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, M_k_z_grid_points)

        '''
        ############################################################################################################

        # compute grid quantities with parallel processing

        ## NOTE: one needs to add a new function called pk_tilt_nonlinear in classy.pyx file of class_public/python to get the logarithmic derivative wrt nonlinear Pk
        #self.n_eff_nl_k_z = function_interp2d_RectBivariateSpline_grid(self.k_grid_points_ascending, self.z_grid_points_ascending, cclass.pk_nonlinear_tilt)
        ####self.n_eff_nl_k_z = function_interp2d_RectBivariateSpline_grid_parallel(self.k_grid_points_ascending, self.z_grid_points_ascending, cclass.pk_nonlinear_tilt)

        global populate_k_z_grid_quantities

        def populate_k_z_grid_quantities(k_and_z):
            k = k_and_z[0]
            z = k_and_z[1]
            
            n_eff = self.n_eff_lin_k_smoothed(k)
            q = k/self.k_nl_z_pk_lin(z) # non-linearity ratio
            sigma8z = self.sigma8_z(z)

            #return [a_GM(n_eff, q, sigma8z), b_GM(n_eff, q), c_GM(n_eff, q), self.compute_RF_G_1_k_z(k, z), self.compute_RF_G_K_k_z(k, z)]
            return [a_GM(n_eff, q, sigma8z), b_GM(n_eff, q), c_GM(n_eff, q), self.compute_RF_G_1_k_z(k, z), self.compute_RF_G_K_k_z(k, z), cclass.pk_nonlinear_tilt(k,z)]

        k_and_z_list = list(itertools.product(self.k_grid_points_ascending, self.z_grid_points_ascending))

        if (constants._compute_R_z_grid_ == True):
            ### tabulate quantities for sigma and its derivatives in R,z grid (parallel)
            global populate_R_z_grid_quantities

            def populate_R_z_grid_quantities(R_and_z):
                R = R_and_z[0]
                z = R_and_z[1]

                return [cclass.sigma(R, z), cclass.sigma_squared_prime(R, z), cclass.sigma_prime(R, z)]

            self.R_grid_points_ascending = np.logspace(np.log10(constants._R_min_), np.log10(constants._R_max_), 100)
            #self.R_grid_points_ascending = np.arange(constants._R_min_, constants._R_max_, step=0.1)
            R_and_z_list = list(itertools.product(self.R_grid_points_ascending, self.z_grid_points_ascending))

            #pool = mp.Pool(processes=mp.cpu_count()-2)
            pool = mp.Pool(processes=constants._num_of_prcoesses_)
            result_k_z = pool.map(populate_k_z_grid_quantities, k_and_z_list)
            result_k_z = np.reshape(np.array(result_k_z, dtype=object), (self.k_grid_points_ascending.size, self.z_grid_points_ascending.size, 6))
            result_R_z = pool.map(populate_R_z_grid_quantities, R_and_z_list)
            result_R_z = np.reshape(np.array(result_R_z, dtype=object), (self.R_grid_points_ascending.size, self.z_grid_points_ascending.size, 3))
            pool.close()
            pool.join()

        else:
            #pool = mp.Pool(processes=mp.cpu_count()-2)
            pool = mp.Pool(processes=constants._num_of_prcoesses_)
            result_k_z = pool.map(populate_k_z_grid_quantities, k_and_z_list)
            result_k_z = np.reshape(np.array(result_k_z, dtype=object), (self.k_grid_points_ascending.size, self.z_grid_points_ascending.size, 6))
            pool.close()
            pool.join()

        B3D_a_GM_k_z_grid_points = result_k_z[:,:,0]
        B3D_b_GM_k_z_grid_points = result_k_z[:,:,1]
        B3D_c_GM_k_z_grid_points = result_k_z[:,:,2]
        RF_G_1_k_z_grid_points   = result_k_z[:,:,3]
        RF_G_K_k_z_grid_points   = result_k_z[:,:,4]
        n_eff_nl_k_z_grid_points = result_k_z[:,:,5]

        self.B3D_a_GM_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, B3D_a_GM_k_z_grid_points)
        self.B3D_b_GM_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, B3D_b_GM_k_z_grid_points)
        self.B3D_c_GM_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, B3D_c_GM_k_z_grid_points)
        self.RF_G_1_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, RF_G_1_k_z_grid_points)
        self.RF_G_K_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, RF_G_K_k_z_grid_points)
        self.n_eff_nl_k_z = interpolate.RectBivariateSpline(self.k_grid_points_ascending, self.z_grid_points_ascending, n_eff_nl_k_z_grid_points)

        if (constants._compute_R_z_grid_ == True):
            sigma_R_z_grid_points = result_R_z[:,:,0]
            sigma_squared_prime_R_z_grid_points = result_R_z[:,:,1]
            sigma_prime_R_z_grid_points = result_R_z[:,:,2]

            self.sigma_R_z = interpolate.RectBivariateSpline(self.R_grid_points_ascending, self.z_grid_points_ascending, sigma_R_z_grid_points)
            self.sigma_squared_prime_R_z = interpolate.RectBivariateSpline(self.R_grid_points_ascending, self.z_grid_points_ascending, sigma_squared_prime_R_z_grid_points)
            self.sigma_prime_R_z = interpolate.RectBivariateSpline(self.R_grid_points_ascending, self.z_grid_points_ascending, sigma_prime_R_z_grid_points)

        ############################################################################################################
        '''

    def sigma8_z(self, z):
        return self.D_plus_z(z)*self.sigma8_0

    def compute_k_nl_z_pk_lin(self, z):
        def f(k):
            return k**3 * self.P3D_k_z_lin(k, z, grid=True)[:,0] - 2*np.pi**2
        return fsolve(f, x0=1.0)

    def P3D_k_z(self, k_arr, z, use_pk_nl=True):
        # Evaluate the 3D power spectrum for a given k array at a single redshift

        vals = np.zeros(k_arr.size)

        if (z < self.z_grid_points_ascending[0] or z > self.z_grid_points_ascending[-1]):
            return vals

        k_arr_valid_mask = (k_arr >= self.k_grid_points_ascending[0]) & (k_arr <= self.k_grid_points_ascending[-1]) # create mask for k array
            
        k_arr_valid = k_arr[k_arr_valid_mask] # valid k's

        if (k_arr_valid.size == 0):
            return vals

        valid_vals = np.zeros(k_arr_valid.size) # create a zeros 1D array for storing the computation of P3D on the valid k values
        k_arr_valid_unsorted_index_order = np.argsort(k_arr_valid) # unsorted array indices of valid k's before they are sorted in ascending order
        k_arr_valid_ascending = k_arr_valid[k_arr_valid_unsorted_index_order] # valid k's sorted in ascending order

        # store the computed Pk at the indices of the unsorted indices of valid k values
        if(use_pk_nl==True):
            valid_vals[k_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k_arr_valid_ascending, z, grid=True)[:,0]
        elif (use_pk_nl==False):
            valid_vals[k_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k_arr_valid_ascending, z, grid=True)[:,0]

        vals[k_arr_valid_mask] = valid_vals

        return vals

    """
    def B3D_tree_k1_k2_k3_arr_z_interpolated(self, k1_arr, k2_arr, k3_arr, z):
        if (self.B3D_tree_grid_computed == False):
            self.B3D_grid_points_ascending = (self.k_grid_points_ascending, self.k_grid_points_ascending, self.k_grid_points_ascending, self.z_grid_points_ascending)
            kk1, kk2, kk3, zz = np.meshgrid(*self.B3D_grid_points_ascending, indexing='ij')
            self.B3D_tree_grid_points_values = B_tree(kk1, kk2, kk3, zz, self.P3D_k_z_lin, {"grid" : False})
            self.B3D_tree_grid_computed = True

        points = np.zeros([k1_arr.size,4])
        points[:,0] = k1_arr
        points[:,1] = k2_arr
        points[:,2] = k3_arr
        points[:,3] = np.ones(k1_arr.size)*z
        return interpolate.interpn(self.B3D_grid_points_ascending, self.B3D_tree_grid_points_values, points, fill_value=0.0)
    """

    def B3D_k1_k2_k3_z(self, k1, k2, k3, z, B3D_type):
        # Evaluate the 3D bispectrum for given k arrays at a single redshift

        vals = np.zeros(k1.size)

        if (z < self.z_grid_points_ascending[0] or z > self.z_grid_points_ascending[-1]):
            return vals

        # arrange each triangle of the bispectrum into hard, medium and soft modes and check whether triangle is closed or not
        k_3D_arr = np.array([k1, k2, k3]).T
        k_3D_arr.sort(axis=1)

        k1_arr = k_3D_arr[:,0] # shortest side (soft)
        k2_arr = k_3D_arr[:,1]
        k3_arr = k_3D_arr[:,2] # longest side (hard)

        valid_triangle_mask = np.array(np.ones(k1.size), dtype=bool) # array of True values  
        valid_triangle_mask[k3_arr > k1_arr + k2_arr] = False # see whether the triangle closes or not (i.e. fulfills triangle inequality)

        k1_arr_valid_mask = (k1_arr >= self.k_grid_points_ascending[0]) & (k1_arr <= self.k_grid_points_ascending[-1]) # create mask for k1 array
        k2_arr_valid_mask = (k2_arr >= self.k_grid_points_ascending[0]) & (k2_arr <= self.k_grid_points_ascending[-1]) # create mask for k2 array
        k3_arr_valid_mask = (k3_arr >= self.k_grid_points_ascending[0]) & (k3_arr <= self.k_grid_points_ascending[-1]) # create mask for k3 array
        k_arr_valid_mask = ( valid_triangle_mask & k1_arr_valid_mask & k2_arr_valid_mask & k3_arr_valid_mask )

        k1_arr_valid = k1_arr[k_arr_valid_mask] # valid k1's
        k2_arr_valid = k2_arr[k_arr_valid_mask] 
        k3_arr_valid = k3_arr[k_arr_valid_mask] 

        if (k1_arr_valid.size == 0):
            return vals

        # now work with only the valid k arrays
        # ------------------------------------------

        valid_vals = np.zeros(k1_arr_valid.size) # create a zeros 1D array for storing the computation of the B3D on the valid ki values

        # for matter bispectrum
        if (B3D_type=="B_tree" or B3D_type=="B_GM" or B3D_type=="B_GMRF"):

            if (B3D_type=="B_tree" or B3D_type=="B_GM"):

                k1_arr_valid_unsorted_index_order = np.argsort(k1_arr_valid) # unsorted array indices of valid k1's before they are sorted in ascending order
                k1_arr_valid_ascending = k1_arr_valid[k1_arr_valid_unsorted_index_order] # valid k1's sorted in ascending order

                k2_arr_valid_unsorted_index_order = np.argsort(k2_arr_valid) 
                k2_arr_valid_ascending = k2_arr_valid[k2_arr_valid_unsorted_index_order] 

                k3_arr_valid_unsorted_index_order = np.argsort(k3_arr_valid)
                k3_arr_valid_ascending = k3_arr_valid[k3_arr_valid_unsorted_index_order]
            
                Pk1_arr_valid = np.zeros(k1_arr_valid.size)
                Pk2_arr_valid = np.zeros(k2_arr_valid.size)
                Pk3_arr_valid = np.zeros(k3_arr_valid.size)

                if (B3D_type=="B_tree"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]
                    
                    valid_vals = B_tree(k1_arr_valid, k2_arr_valid, k3_arr_valid, 
                                        Pk1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)
                    
                elif (B3D_type=="B_GM"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]

                    a1_arr_valid = np.zeros(k1_arr_valid.size)
                    a2_arr_valid = np.zeros(k2_arr_valid.size)
                    a3_arr_valid = np.zeros(k3_arr_valid.size)

                    a1_arr_valid[k1_arr_valid_unsorted_index_order] = self.B3D_a_GM_k_z(k1_arr_valid_ascending, z, grid=True)[:,0]
                    a2_arr_valid[k2_arr_valid_unsorted_index_order] = self.B3D_a_GM_k_z(k2_arr_valid_ascending, z, grid=True)[:,0]
                    a3_arr_valid[k3_arr_valid_unsorted_index_order] = self.B3D_a_GM_k_z(k3_arr_valid_ascending, z, grid=True)[:,0]

                    b1_arr_valid = np.zeros(k1_arr_valid.size)
                    b2_arr_valid = np.zeros(k2_arr_valid.size)
                    b3_arr_valid = np.zeros(k3_arr_valid.size)

                    b1_arr_valid[k1_arr_valid_unsorted_index_order] = self.B3D_b_GM_k_z(k1_arr_valid_ascending, z, grid=True)[:,0]
                    b2_arr_valid[k2_arr_valid_unsorted_index_order] = self.B3D_b_GM_k_z(k2_arr_valid_ascending, z, grid=True)[:,0]
                    b3_arr_valid[k3_arr_valid_unsorted_index_order] = self.B3D_b_GM_k_z(k3_arr_valid_ascending, z, grid=True)[:,0]

                    c1_arr_valid = np.zeros(k1_arr_valid.size)
                    c2_arr_valid = np.zeros(k2_arr_valid.size)
                    c3_arr_valid = np.zeros(k3_arr_valid.size)

                    c1_arr_valid[k1_arr_valid_unsorted_index_order] = self.B3D_c_GM_k_z(k1_arr_valid_ascending, z, grid=True)[:,0]
                    c2_arr_valid[k2_arr_valid_unsorted_index_order] = self.B3D_c_GM_k_z(k2_arr_valid_ascending, z, grid=True)[:,0]
                    c3_arr_valid[k3_arr_valid_unsorted_index_order] = self.B3D_c_GM_k_z(k3_arr_valid_ascending, z, grid=True)[:,0]

                    valid_vals = B_GM(k1_arr_valid, k2_arr_valid, k3_arr_valid,
                                    Pk1_arr_valid, Pk2_arr_valid, Pk3_arr_valid,
                                    a1_arr_valid, a2_arr_valid, a3_arr_valid,
                                    b1_arr_valid, b2_arr_valid, b3_arr_valid, 
                                    c1_arr_valid, c2_arr_valid, c3_arr_valid)

            elif (B3D_type=="B_GMRF"):     

                sq_mask = (k2_arr_valid / k1_arr_valid > self.f_sq)

                # for the squeezed configurations

                k1_arr_valid_sq = k1_arr_valid[sq_mask] # valid k1's in squeezed triangles i.e. ks
                k2_arr_valid_sq = k2_arr_valid[sq_mask] # valid k2's in squeezed triangles i.e. km
                k3_arr_valid_sq = k3_arr_valid[sq_mask] # valid k3's in squeezed triangles i.e. kh 

                k1_arr_valid_sq_unsorted_index_order = np.argsort(k1_arr_valid_sq) # unsorted array indices of valid k1's in squeezed triangles before they are sorted in ascending order
                k1_arr_valid_sq_ascending = k1_arr_valid_sq[k1_arr_valid_sq_unsorted_index_order] # valid k1's in squeezed triangles sorted in ascending order
                
                #k2_arr_valid_sq_unsorted_index_order = np.argsort(k2_arr_valid_sq)
                #k2_arr_valid_sq_ascending = k2_arr_valid_sq[k2_arr_valid_sq_unsorted_index_order]

                k3_arr_valid_sq_unsorted_index_order = np.argsort(k3_arr_valid_sq)
                k3_arr_valid_sq_ascending = k3_arr_valid_sq[k3_arr_valid_sq_unsorted_index_order]

                Pk1_arr_valid_sq = np.zeros(k1_arr_valid_sq.size)
                #Pk2_arr_valid_sq = np.zeros(k2_arr_valid_sq.size)
                Pk3_arr_valid_sq = np.zeros(k3_arr_valid_sq.size)

                Pk1_arr_valid_sq[k1_arr_valid_sq_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_sq_ascending, z, grid=True)[:,0]
                #Pk2_arr_valid_sq[k2_arr_valid_sq_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_sq_ascending, z, grid=True)[:,0]
                Pk3_arr_valid_sq[k3_arr_valid_sq_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_sq_ascending, z, grid=True)[:,0]

                n_h_arr_valid_sq = np.zeros(k3_arr_valid_sq.size)
                G_1_h_arr_valid_sq = np.zeros(k3_arr_valid_sq.size)
                G_K_h_arr_valid_sq = np.zeros(k3_arr_valid_sq.size)

                n_h_arr_valid_sq[k3_arr_valid_sq_unsorted_index_order] = self.n_eff_nl_k_z(k3_arr_valid_sq_ascending, z, grid=True)[:,0]
                G_1_h_arr_valid_sq[k3_arr_valid_sq_unsorted_index_order] = self.RF_G_1_k_z(k3_arr_valid_sq_ascending, z, grid=True)[:,0]
                G_K_h_arr_valid_sq[k3_arr_valid_sq_unsorted_index_order] = self.RF_G_K_k_z(k3_arr_valid_sq_ascending, z, grid=True)[:,0]

                valid_vals[sq_mask] = B_RF(k1_arr_valid_sq, k2_arr_valid_sq, k3_arr_valid_sq, 
                                        Pk1_arr_valid_sq, Pk3_arr_valid_sq, 
                                        n_h_arr_valid_sq, G_1_h_arr_valid_sq, G_K_h_arr_valid_sq)

                # for the non-squeezed configurations 

                k1_arr_valid_nonsq = k1_arr_valid[~sq_mask] # valid k1's in non-squeezed triangles
                k2_arr_valid_nonsq = k2_arr_valid[~sq_mask] 
                k3_arr_valid_nonsq = k3_arr_valid[~sq_mask] 

                k1_arr_valid_nonsq_unsorted_index_order = np.argsort(k1_arr_valid_nonsq) # unsorted array indices of valid k1's in non-squeezed triangles before they are sorted in ascending order
                k1_arr_valid_nonsq_ascending = k1_arr_valid_nonsq[k1_arr_valid_nonsq_unsorted_index_order] # valid k1's in non-squeezed triangles sorted in ascending order
                
                k2_arr_valid_nonsq_unsorted_index_order = np.argsort(k2_arr_valid_nonsq)
                k2_arr_valid_nonsq_ascending = k2_arr_valid_nonsq[k2_arr_valid_nonsq_unsorted_index_order]

                k3_arr_valid_nonsq_unsorted_index_order = np.argsort(k3_arr_valid_nonsq)
                k3_arr_valid_nonsq_ascending = k3_arr_valid_nonsq[k3_arr_valid_nonsq_unsorted_index_order]

                Pk1_arr_valid_nonsq = np.zeros(k1_arr_valid_nonsq.size)
                Pk2_arr_valid_nonsq = np.zeros(k2_arr_valid_nonsq.size)
                Pk3_arr_valid_nonsq = np.zeros(k3_arr_valid_nonsq.size)

                Pk1_arr_valid_nonsq[k1_arr_valid_nonsq_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_nonsq_ascending, z, grid=True)[:,0]
                Pk2_arr_valid_nonsq[k2_arr_valid_nonsq_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_nonsq_ascending, z, grid=True)[:,0]
                Pk3_arr_valid_nonsq[k3_arr_valid_nonsq_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_nonsq_ascending, z, grid=True)[:,0]

                a1_arr_valid_nonsq = np.zeros(k1_arr_valid_nonsq.size)
                a2_arr_valid_nonsq = np.zeros(k2_arr_valid_nonsq.size)
                a3_arr_valid_nonsq = np.zeros(k3_arr_valid_nonsq.size)

                a1_arr_valid_nonsq[k1_arr_valid_nonsq_unsorted_index_order] = self.B3D_a_GM_k_z(k1_arr_valid_nonsq_ascending, z, grid=True)[:,0]
                a2_arr_valid_nonsq[k2_arr_valid_nonsq_unsorted_index_order] = self.B3D_a_GM_k_z(k2_arr_valid_nonsq_ascending, z, grid=True)[:,0]
                a3_arr_valid_nonsq[k3_arr_valid_nonsq_unsorted_index_order] = self.B3D_a_GM_k_z(k3_arr_valid_nonsq_ascending, z, grid=True)[:,0]

                b1_arr_valid_nonsq = np.zeros(k1_arr_valid_nonsq.size)
                b2_arr_valid_nonsq = np.zeros(k2_arr_valid_nonsq.size)
                b3_arr_valid_nonsq = np.zeros(k3_arr_valid_nonsq.size)

                b1_arr_valid_nonsq[k1_arr_valid_nonsq_unsorted_index_order] = self.B3D_b_GM_k_z(k1_arr_valid_nonsq_ascending, z, grid=True)[:,0]
                b2_arr_valid_nonsq[k2_arr_valid_nonsq_unsorted_index_order] = self.B3D_b_GM_k_z(k2_arr_valid_nonsq_ascending, z, grid=True)[:,0]
                b3_arr_valid_nonsq[k3_arr_valid_nonsq_unsorted_index_order] = self.B3D_b_GM_k_z(k3_arr_valid_nonsq_ascending, z, grid=True)[:,0]

                c1_arr_valid_nonsq = np.zeros(k1_arr_valid_nonsq.size)
                c2_arr_valid_nonsq = np.zeros(k2_arr_valid_nonsq.size)
                c3_arr_valid_nonsq = np.zeros(k3_arr_valid_nonsq.size)

                c1_arr_valid_nonsq[k1_arr_valid_nonsq_unsorted_index_order] = self.B3D_c_GM_k_z(k1_arr_valid_nonsq_ascending, z, grid=True)[:,0]
                c2_arr_valid_nonsq[k2_arr_valid_nonsq_unsorted_index_order] = self.B3D_c_GM_k_z(k2_arr_valid_nonsq_ascending, z, grid=True)[:,0]
                c3_arr_valid_nonsq[k3_arr_valid_nonsq_unsorted_index_order] = self.B3D_c_GM_k_z(k3_arr_valid_nonsq_ascending, z, grid=True)[:,0]

                valid_vals[~sq_mask] = B_GM(k1_arr_valid_nonsq, k2_arr_valid_nonsq, k3_arr_valid_nonsq, 
                                            Pk1_arr_valid_nonsq, Pk2_arr_valid_nonsq, Pk3_arr_valid_nonsq,
                                            a1_arr_valid_nonsq, a2_arr_valid_nonsq, a3_arr_valid_nonsq,
                                            b1_arr_valid_nonsq, b2_arr_valid_nonsq, b3_arr_valid_nonsq, 
                                            c1_arr_valid_nonsq, c2_arr_valid_nonsq, c3_arr_valid_nonsq)

            #if (constants._apply_T17_corrections_ == True):
            #    vals = vals*T17_shell_correction(k1_arr/self.h)*T17_shell_correction(k2_arr/self.h)*T17_shell_correction(k3_arr/self.h)

            # primordial non-Gaussianity to matter-bispectrum
            if (self.f_NL != 0):
                
                k1_arr_valid_unsorted_index_order = np.argsort(k1_arr_valid) # unsorted array indices of valid k1's before they are sorted in ascending order
                k1_arr_valid_ascending = k1_arr_valid[k1_arr_valid_unsorted_index_order] # valid k1's sorted in ascending order

                k2_arr_valid_unsorted_index_order = np.argsort(k2_arr_valid) 
                k2_arr_valid_ascending = k2_arr_valid[k2_arr_valid_unsorted_index_order] 

                k3_arr_valid_unsorted_index_order = np.argsort(k3_arr_valid)
                k3_arr_valid_ascending = k3_arr_valid[k3_arr_valid_unsorted_index_order]
            
                Pk1_arr_valid = np.zeros(k1_arr_valid.size)
                Pk2_arr_valid = np.zeros(k2_arr_valid.size)
                Pk3_arr_valid = np.zeros(k3_arr_valid.size)

                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]

                Mk1_arr_valid = np.zeros(k1_arr_valid.size)
                Mk2_arr_valid = np.zeros(k2_arr_valid.size)
                Mk3_arr_valid = np.zeros(k3_arr_valid.size)

                Mk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.M_k_z(k1_arr_valid_ascending, z, grid=True)[:,0]
                Mk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.M_k_z(k2_arr_valid_ascending, z, grid=True)[:,0]
                Mk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.M_k_z(k3_arr_valid_ascending, z, grid=True)[:,0]

                if (self.f_NL_type=="local"):
                    valid_vals += B_primordial_local(self.f_NL, Mk1_arr_valid, Mk2_arr_valid, Mk3_arr_valid, Pk1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)
                elif (self.f_NL_type=="equilateral"):
                    valid_vals += B_primordial_equilateral(self.f_NL, Mk1_arr_valid, Mk2_arr_valid, Mk3_arr_valid, Pk1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)
                elif (self.f_NL_type=="orthogonal"):
                    valid_vals += B_primordial_orthogonal(self.f_NL, Mk1_arr_valid, Mk2_arr_valid, Mk3_arr_valid, Pk1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)
                
        # for galaxy-matter bispectrum
        else:

            k1_arr_valid_unsorted_index_order = np.argsort(k1_arr_valid) # unsorted array indices of valid k1's before they are sorted in ascending order
            k1_arr_valid_ascending = k1_arr_valid[k1_arr_valid_unsorted_index_order] # valid k1's sorted in ascending order

            k2_arr_valid_unsorted_index_order = np.argsort(k2_arr_valid) 
            k2_arr_valid_ascending = k2_arr_valid[k2_arr_valid_unsorted_index_order] 

            k3_arr_valid_unsorted_index_order = np.argsort(k3_arr_valid)
            k3_arr_valid_ascending = k3_arr_valid[k3_arr_valid_unsorted_index_order]
        
            Pk1_arr_valid = np.zeros(k1_arr_valid.size)
            Pk2_arr_valid = np.zeros(k2_arr_valid.size)
            Pk3_arr_valid = np.zeros(k3_arr_valid.size)

            if (B3D_type=="B_gmm_b2_lin" or B3D_type=="B_mgm_b2_lin" or B3D_type=="B_mmg_b2_lin"):
                if (B3D_type=="B_gmm_b2_lin"):
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_PaPb(Pk2_arr_valid, Pk3_arr_valid)
                elif (B3D_type=="B_mgm_b2_lin"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_PaPb(Pk1_arr_valid, Pk3_arr_valid)
                elif (B3D_type=="B_mmg_b2_lin"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_PaPb(Pk1_arr_valid, Pk2_arr_valid)

            elif (B3D_type=="B_gmm_bs2_lin" or B3D_type=="B_mgm_bs2_lin" or B3D_type=="B_mmg_bs2_lin"):
                if (B3D_type=="B_gmm_bs2_lin"):
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_S2PaPb(k2_arr_valid, k3_arr_valid, k1_arr_valid, Pk2_arr_valid, Pk3_arr_valid) 
                elif (B3D_type=="B_mgm_bs2_lin"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_S2PaPb(k1_arr_valid, k3_arr_valid, k2_arr_valid, Pk1_arr_valid, Pk3_arr_valid)     
                elif (B3D_type=="B_mmg_bs2_lin"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_S2PaPb(k1_arr_valid, k2_arr_valid, k3_arr_valid, Pk1_arr_valid, Pk2_arr_valid)     

            elif (B3D_type=="B_ggm_b1_b2_lin" or B3D_type=="B_gmg_b1_b2_lin" or B3D_type=="B_mgg_b1_b2_lin"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]

                if (B3D_type=="B_ggm_b1_b2_lin"):
                    valid_vals = B_PaPb(Pk1_arr_valid, Pk3_arr_valid) + B_PaPb(Pk2_arr_valid, Pk3_arr_valid)      
                elif (B3D_type=="B_gmg_b1_b2_lin"):
                    valid_vals = B_PaPb(Pk1_arr_valid, Pk2_arr_valid) + B_PaPb(Pk3_arr_valid, Pk2_arr_valid)   
                elif (B3D_type=="B_mgg_b1_b2_lin"):
                    valid_vals = B_PaPb(Pk2_arr_valid, Pk1_arr_valid) + B_PaPb(Pk3_arr_valid, Pk1_arr_valid)   

            elif (B3D_type=="B_ggm_b1_bs2_lin" or B3D_type=="B_gmg_b1_bs2_lin" or B3D_type=="B_mgg_b1_bs2_lin"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]

                if (B3D_type=="B_ggm_b1_bs2_lin"):
                    valid_vals = B_S2PaPb(k1_arr_valid, k3_arr_valid, k2_arr_valid, Pk1_arr_valid, Pk3_arr_valid) +  B_S2PaPb(k2_arr_valid, k3_arr_valid, k1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)     
                elif (B3D_type=="B_gmg_b1_bs2_lin"):
                    valid_vals = B_S2PaPb(k1_arr_valid, k2_arr_valid, k3_arr_valid, Pk1_arr_valid, Pk2_arr_valid) +  B_S2PaPb(k3_arr_valid, k2_arr_valid, k1_arr_valid, Pk3_arr_valid, Pk2_arr_valid)     
                elif (B3D_type=="B_mgg_b1_bs2_lin"):
                    valid_vals = B_S2PaPb(k2_arr_valid, k1_arr_valid, k3_arr_valid, Pk2_arr_valid, Pk1_arr_valid) +  B_S2PaPb(k3_arr_valid, k1_arr_valid, k2_arr_valid, Pk3_arr_valid, Pk1_arr_valid)     

            elif (B3D_type=="B_ggm_eps_epsdelta_lin" or B3D_type=="B_gmg_eps_epsdelta_lin" or B3D_type=="B_mgg_eps_epsdelta_lin"):
                if (B3D_type=="B_ggm_eps_epsdelta_lin"):
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = 2*Pk3_arr_valid
                elif (B3D_type=="B_gmg_eps_epsdelta_lin"):
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = 2*Pk2_arr_valid
                elif (B3D_type=="B_mgg_eps_epsdelta_lin"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = 2*Pk1_arr_valid

            elif (B3D_type=="B_ggg_b1sq_b2_lin"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]
                valid_vals = B_PP(Pk1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)     

            elif (B3D_type=="B_ggg_b1sq_bs2_lin"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]
                valid_vals = B_S2PP(k1_arr_valid, k2_arr_valid, k3_arr_valid, Pk1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)      

            elif (B3D_type=="B_ggg_b1_eps_epsdelta_lin"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_lin(k3_arr_valid_ascending, z, grid=True)[:,0]
                valid_vals = 2*(Pk1_arr_valid + Pk2_arr_valid + Pk3_arr_valid)

            elif (B3D_type=="B_gmm_b2_nl" or B3D_type=="B_mgm_b2_nl" or B3D_type=="B_mmg_b2_nl"):
                if (B3D_type=="B_gmm_b2_nl"):
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_PaPb(Pk2_arr_valid, Pk3_arr_valid)
                elif (B3D_type=="B_mgm_b2_nl"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_PaPb(Pk1_arr_valid, Pk3_arr_valid)
                elif (B3D_type=="B_mmg_b2_nl"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_PaPb(Pk1_arr_valid, Pk2_arr_valid)

            elif (B3D_type=="B_gmm_bs2_nl" or B3D_type=="B_mgm_bs2_nl" or B3D_type=="B_mmg_bs2_nl"):
                if (B3D_type=="B_gmm_bs2_nl"):
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_S2PaPb(k2_arr_valid, k3_arr_valid, k1_arr_valid, Pk2_arr_valid, Pk3_arr_valid) 
                elif (B3D_type=="B_mgm_bs2_nl"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_S2PaPb(k1_arr_valid, k3_arr_valid, k2_arr_valid, Pk1_arr_valid, Pk3_arr_valid)     
                elif (B3D_type=="B_mmg_bs2_nl"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = B_S2PaPb(k1_arr_valid, k2_arr_valid, k3_arr_valid, Pk1_arr_valid, Pk2_arr_valid)     

            elif (B3D_type=="B_ggm_b1_b2_nl" or B3D_type=="B_gmg_b1_b2_nl" or B3D_type=="B_mgg_b1_b2_nl"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]

                if (B3D_type=="B_ggm_b1_b2_nl"):
                    valid_vals = B_PaPb(Pk1_arr_valid, Pk3_arr_valid) + B_PaPb(Pk2_arr_valid, Pk3_arr_valid)      
                elif (B3D_type=="B_gmg_b1_b2_nl"):
                    valid_vals = B_PaPb(Pk1_arr_valid, Pk2_arr_valid) + B_PaPb(Pk3_arr_valid, Pk2_arr_valid)   
                elif (B3D_type=="B_mgg_b1_b2_nl"):
                    valid_vals = B_PaPb(Pk2_arr_valid, Pk1_arr_valid) + B_PaPb(Pk3_arr_valid, Pk1_arr_valid)   

            elif (B3D_type=="B_ggm_b1_bs2_nl" or B3D_type=="B_gmg_b1_bs2_nl" or B3D_type=="B_mgg_b1_bs2_nl"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]

                if (B3D_type=="B_ggm_b1_bs2_nl"):
                    valid_vals = B_S2PaPb(k1_arr_valid, k3_arr_valid, k2_arr_valid, Pk1_arr_valid, Pk3_arr_valid) +  B_S2PaPb(k2_arr_valid, k3_arr_valid, k1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)     
                elif (B3D_type=="B_gmg_b1_bs2_nl"):
                    valid_vals = B_S2PaPb(k1_arr_valid, k2_arr_valid, k3_arr_valid, Pk1_arr_valid, Pk2_arr_valid) +  B_S2PaPb(k3_arr_valid, k2_arr_valid, k1_arr_valid, Pk3_arr_valid, Pk2_arr_valid)     
                elif (B3D_type=="B_mgg_b1_bs2_nl"):
                    valid_vals = B_S2PaPb(k2_arr_valid, k1_arr_valid, k3_arr_valid, Pk2_arr_valid, Pk1_arr_valid) +  B_S2PaPb(k3_arr_valid, k1_arr_valid, k2_arr_valid, Pk3_arr_valid, Pk1_arr_valid)     

            elif (B3D_type=="B_ggm_eps_epsdelta_nl" or B3D_type=="B_gmg_eps_epsdelta_nl" or B3D_type=="B_mgg_eps_epsdelta_nl"):
                if (B3D_type=="B_ggm_eps_epsdelta_nl"):
                    Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = 2*Pk3_arr_valid
                elif (B3D_type=="B_gmg_eps_epsdelta_nl"):
                    Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = 2*Pk2_arr_valid
                elif (B3D_type=="B_mgg_eps_epsdelta_nl"):
                    Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                    valid_vals = 2*Pk1_arr_valid

            elif (B3D_type=="B_ggg_b1sq_b2_nl"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]
                valid_vals = B_PP(Pk1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)     

            elif (B3D_type=="B_ggg_b1sq_bs2_nl"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]
                valid_vals = B_S2PP(k1_arr_valid, k2_arr_valid, k3_arr_valid, Pk1_arr_valid, Pk2_arr_valid, Pk3_arr_valid)      

            elif (B3D_type=="B_ggg_b1_eps_epsdelta_nl"):
                Pk1_arr_valid[k1_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k1_arr_valid_ascending, z, grid=True)[:,0]
                Pk2_arr_valid[k2_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k2_arr_valid_ascending, z, grid=True)[:,0]
                Pk3_arr_valid[k3_arr_valid_unsorted_index_order] = self.P3D_k_z_nl(k3_arr_valid_ascending, z, grid=True)[:,0]
                valid_vals = 2*(Pk1_arr_valid + Pk2_arr_valid + Pk3_arr_valid)

            elif (B3D_type=="B_ggg_eps_eps_eps"):
                valid_vals = 1

        vals[k_arr_valid_mask] = valid_vals

        #if (constants._apply_T17_corrections_ == True):
        #    vals = vals*T17_shell_correction(k1_arr/self.h)*T17_shell_correction(k2_arr/self.h)*T17_shell_correction(k3_arr/self.h)

        return vals

    def compute_RF_G_1_k_z(self, k, z):
        if (self.RF_G_1_table_data_loaded == False):
            RF_G_1_k_table_data = np.loadtxt('../data/response_functions/G_1_k_h.tab', usecols=[0])*self.h
            RF_G_1_z_table_data = np.loadtxt('../data/response_functions/G_1_z.tab', usecols=[0])
            RF_G_1_k_z_table_data = np.loadtxt('../data/response_functions/G_1_vals.tab', usecols=[0]).reshape(RF_G_1_k_table_data.size, RF_G_1_z_table_data.size)

            self.RF_G_1_k_z_table = interpolate.RectBivariateSpline(RF_G_1_k_table_data, RF_G_1_z_table_data, RF_G_1_k_z_table_data)
            self.RF_G_1_k_min_table_data = RF_G_1_k_table_data[0]
            self.RF_G_1_k_max_table_data = RF_G_1_k_table_data[-1]
            self.RF_G_1_z_max_table_data = RF_G_1_z_table_data[-1]
            self.RF_G_1_table_data_loaded = True

        #if (z > self.RF_G_1_z_max_table_data):
        #    return 0.0

        # NOTE: if z > self.RF_G_1_z_max_table_data ; just set it in accordance with the value at the last redshift in the table

        k_min = self.RF_G_1_k_min_table_data
        k_max = self.RF_G_1_k_max_table_data

        if (k < k_min):
            return 26.0/21.0
        elif (k > k_max):
            B_1 = -3.0/4.0
            return B_1 + (self.RF_G_1_k_z_table(k_max, z)-B_1)*np.sqrt(k_max/k)
        else:
            return self.RF_G_1_k_z_table(k, z)

    def compute_RF_G_K_k_z(self, k, z):
        if (self.RF_G_K_table_data_loaded == False):
            RF_G_K_k_table_data = np.loadtxt('../data/response_functions/G_K_k_h.tab', usecols=[0])*self.h
            RF_G_K_z_table_data = np.loadtxt('../data/response_functions/G_K_z.tab', usecols=[0])
            RF_G_K_k_z_table_data = np.loadtxt('../data/response_functions/G_K_vals.tab', usecols=[0]).reshape(RF_G_K_k_table_data.size, RF_G_K_z_table_data.size)

            self.RF_G_K_k_z_table = interpolate.RectBivariateSpline(RF_G_K_k_table_data, RF_G_K_z_table_data, RF_G_K_k_z_table_data)
            self.RF_G_K_k_min_table_data = RF_G_K_k_table_data[0]
            self.RF_G_K_k_max_table_data = RF_G_K_k_table_data[-1]
            self.RF_G_K_z_max_table_data = RF_G_K_z_table_data[-1]
            self.RF_G_K_table_data_loaded = True

        #if (z > self.RF_G_K_z_max_table_data):
        #    return 0.0

        # NOTE: if z > self.RF_G_K_z_max_table_data ; just set it in accordance with the value at the last redshift in the table

        k_min = self.RF_G_K_k_min_table_data
        k_max = self.RF_G_K_k_max_table_data

        if (k < k_min):
            return 8.0/7.0
        elif (k > k_max):
            B_K = -9.0/4.0
            return B_K + (self.RF_G_K_k_z_table(k_max, z)-B_K)*np.sqrt(k_max/k)
        else:
            return self.RF_G_K_k_z_table(k, z)
        
    def compute_M_k_z(self, k, z):
        return 2*k**2/(3*self.Omega0_m*self.H_z(0)**2)*self.T_k(k)*self.D_z(z)

def T17_shell_correction(k_h_over_Mpc):
    # finite lens-shell-thickness effect in T17

    return np.sqrt( (1.0+constants._c1_T17_*k_h_over_Mpc**(-constants._a1_T17_))**constants._a1_T17_ / (1.0+constants._c2_T17_*k_h_over_Mpc**(-constants._a2_T17_))**constants._a3_T17_ )

def T17_resolution_correction(ell):
    # finite angular resolution healpy map effect in T17

    return 1.0 / ( 1.0 + (ell/constants._ell_res_T17_)**2)