# i3PCF
**The integrated 3PCF of projected cosmic density fields**

This repository hosts code for computing theoretical predictions for the integrated 3-point correlation function of projected cosmic density fields. It also computes predictions for the 2-point correlation function of projected fields. 

## Repository structure

- **/data/** folder contains pre-tabulated data (such as redshift distributions of galaxies etc.) that must be loaded and used in the code.
- **/helper/** folder contains some helper files (e.g. file containing functions which need to be copied to classy.pyx).
- **/output/** folder contains the paths in which the code's output (e.g. spectra, correlation functions, etc.) will be saved.
- **/src/** folder contains the files for performing the computations. The base file is main.py which performs all the computations.

## Installation

For a clean installation, first deactivate your current environment and then create a new environment (e.g. name i3PCF) with conda 

```
conda deactivate
conda create -n i3PCF python=3
```

Once created, activate the environment

```
conda activate i3PCF
```

We need the following dependencies for running the i3PCF code:

- [**numpy**](https://pypi.org/project/numpy/) 
- [**scipy**](https://pypi.org/project/scipy/) 
- [**matplotlib**](https://pypi.org/project/matplotlib/)
- [**pandas**](https://pypi.org/project/pandas/)
- [**vegas**](https://pypi.org/project/vegas/) (Monte-Carlo integration)
- [**treecorr**](https://pypi.org/project/TreeCorr/) (for computing the binning of the correlation functions)
- [**healpy**](https://pypi.org/project/healpy/) (for operations on the sphere)
- [**classy**](https://pypi.org/project/classy/) (Boltzmann solver)

All of these can be installed with pip:

```
pip install numpy scipy matplotlib pandas vegas treecorr healpy classy
```

This should install all the necessary files for running the code!

*Note*: in case the classy installation does not work (possibly due to a classy ctypedef int error) or you need to do some advanced uses of classy (see *Advanced uses* below), first install the other dependencies along with the [**Cython**](https://pypi.org/project/cython/) package:

```
pip install numpy scipy matplotlib pandas vegas treecorr healpy Cython
```

Then go to your home directory and create a folder (if it doesn't exist e.g. call it software) where you can download repositories from github. There, clone the class_public github repository manually

```
cd ~/software
git clone https://github.com/lesgourg/class_public.git
cd class_public/python
```

1. *classy ctypedef int error*: If you faced the following error while installing classy:

'''
Error compiling Cython file:
...
        return d.items()
    else:
        return d.viewitems()

ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_i
         ^

classy.pyx:35:9: 'int_t' is not a type identifier
'''

Then edit the classy.pyx file inside the class_public/python/ folder and change that following line to:

'''
#ctypedef np.int_t DTYPE_i
ctypedef np.int64_t DTYPE_i
'''

Then, execute the following commands:

```
cd ..
make -j
```

This should successfully install classy without the ctypedef error.

2. *Advanced uses*: Or, if you need to do some *advanced uses* with classy, the following tweak to classy.pyx needs to be performed:

Similar to what needed to be done for fixing the *classy ctypedef int error* issue, we will need to recompile class after editing the python/classy.pyx file by adding the following functions

 - sigma_squared_prime
 - sigma_prime
 - pk_nonlinear_tilt

 (please follow the instructions in the file helper/classy_functions_to_add.txt)

## Running the code

- Give your desired input settings in the file **/src/input.py** (e.g. number of tomographic bins, kind of correlations shear x galaxy, which bispectrum recipe to use, etc.)
- Execute the code from inside the **/src/** folder with:

```
python main.py 0 1
```

where the 0 and 1 are needed to tell the code which cosmologies to execute e.g. if you have an array with multiple cosmologies (from Latin-hypercube sampling), then 0 and 1 says start at the 0th index of the array and stop until you reach the 1st index.
 
## Schematic workflow

The main equations that need to be numerically computed to obtain the integrated 3PCF are the following two:

<img width="1213" alt="image" src="https://github.com/anikhalder/i3PCF/assets/20087664/c672411a-216c-414e-b99d-6014227df73b">

where $` \mathcal{B}_{\pm}(\ell) `$ is the integrated bispectrum in 2D and $` \zeta_{\pm}(\alpha) `$ is the corresponding integrated 3PCF obtained on converting the $\mathcal{B}_{\pm}(\ell)$ to a real-space correlation function.

Here is a layout of how the code works to compute the equations above (not all files that are used are shown and only the main pathway of the code is depicted to show how the computation flows):

<img width="1294" alt="image" src="https://github.com/anikhalder/i3PCF/assets/20087664/8beeadea-85b9-4834-aee6-1f6fbc2a9412">


