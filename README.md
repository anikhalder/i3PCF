# i3PCF
**The integrated 3PCF of projected cosmic density fields**

This repository hosts code for computing theoretical predictions for the integrated 3-point correlation function of projected cosmic density fields. It also computes predictions for the 2-point correlation function of projected fields. 

## Repository structure

- **/data/** folder contains pre-tabulated data (such as redshift distributions of galaxies etc.) that must be loaded and used in the code.
- **/output/** folder contains the paths in which the code's output (e.g. spectra, correlation functions, etc.) will be saved.
- **/src/** folder contains the files for performing the computations. The base file is main.py which performs all the computations.

## Dependencies

- **vegas** (Monte-Carlo integration)
- **class** (Boltzmann solver)
- **treecorr** (for computing the binning of the correlation functions)

A requirement before running this code:

Recompile class after editing the python/class.pyx file by adding the following functions.
 - sigma_squared_prime
 - sigma_prime
 - pk_nonlinear_tilt

## Running the code

- Give your desired input settings in the file /src/input.py (e.g. number of tomographic bins, kind of correlations shear x galaxy, which bispectrum recipe to use, etc.)
- Execute the code from inside the /src/ with:

python main.py 0 1

where the 0 and 1 are needed to tell the code which cosmologies to execute e.g. if you have an array with multiple cosmologies (from Latin-hypercube sampling), then 0 and 1 says start at the 0th index of the array and stop until you reach the 1st index.
 
## Schematic workflow

Here is a layout of how the code works (not all files that are used are shown and only the main pathway of the code is depicted to show how the computation flow is performed):

<img width="1294" alt="image" src="https://github.com/anikhalder/i3PCF/assets/20087664/8beeadea-85b9-4834-aee6-1f6fbc2a9412">


