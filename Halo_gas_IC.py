# Generate initial conditions for a DM halo with a spherical gas component (Burkert profile)
# Can be used to model a cluster or a spherical galaxy

# improvements over first version:
# * Integration over logspace
# * number of integration steps independent of r value
# * More importantly, the integral for the pressure is now actually up to 
# * infinity and not just up to r_max

import pynbody as pn
from matplotlib import pyplot as plt
import matplotlib.animation as anim
import numpy as np
from IPython.display import HTML
import warnings
import matplotlib
matplotlib.rcParams['figure.figsize'] = [20,10]
plt.style.use('dark_background')
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import quad
from random import uniform
from random import gauss
import tools_python as tp

############################################################################################################
################################################## SETTINGS ################################################
############################################################################################################

reuse_tables = False # reuse integration tables. Make sure you are using the same parameters

gas_profile = 'Hernquist'
#gas_profile = 'Burkert'

N_sampling = 100000 # size of the arrays created in numerical integration

N_part_gas = 100000 # number of particles to sample
r_s_gas = 194.78 # scale radius
# if rho_s is specified, the total mass will be calculated.
# if not, it will be calculated from the total mass.
rho_s_gas = 9.72e-5 # scale density
#total_mass_gas = 1 # total mass within r_max in 10^10 M_solar

N_part_DM = 100000
r_max = 20 # maximum sampling distance of the system in units of r_s_DM
r_s_DM = 389.31
# if rho_s is specified, the total mass will be calculated.
# if not, it will be calculated from the total mass.
rho_s_DM = 1.14e-4
#total_mass_DM = 100 # total mass within r_max in 10^10 M_solar

############################################################################################################
################################################ FUNCTIONS #################################################
############################################################################################################

def find_closest(array, value):
    # find index of 'array' which is closest to 'value'
    return np.abs(array - value).argmin()

def Burkert(r):
    x = r/r_s_gas
    return rho_s_gas/((1 + x)*(1 + x**2))

def NFW(r):
    x = r/r_s_DM
    return rho_s_DM/(x * (1+x**2))

def Hernquist(r):
    x = r/r_s_DM
    return rho_s_DM/(x * (1+x)**3)

def mass_Hernquist(r):
    # return mass enclosed in r
    x = r/r_s_DM
    return 2 * np.pi * rho_s_DM * r_s_DM**3 * x**2/(x+1)**2

def mass_NFW(r):
    # return mass enclosed in r
    x = r/r_s_DM
    return 2 * np.pi * rho_s_DM * r_s_DM**3 * np.log(x**2 + 1)

def inverse_CPD_NFW(x):
    # draw random radius from inverse CPD for NFW profile.
    # assumes that x is drawn from a uniform distribution 
    # between 0 and 1. 
    # r_max is the maximum sampling distance in units of r_s
    # (has to be finite in this case)
    x *= np.log((r_max/r_s_DM)**2 +1)
    return r_s_DM * np.sqrt(np.exp(x) -1)

def mass_Burkert(r):
    def dMdr(r):
        return 4 * np.pi * Burkert(r) * r**2
    return quad(dMdr, 0, r)[0]

if gas_profile == 'Burkert':
    def density_gas(r):
        return Burkert(r)
    def mass_gas(r):
        return mass_Burkert(r)
elif gas_profile == 'Hernquist':
    def density_gas(r):
        return Hernquist(r)
    def mass_gas(r):
        return mass_Hernquist(r)
else:
    raise Expection("Unknown density profile <",gas_profile,">")

def temp_gas(r):
    def dpdr(r):
        return  43000 * density_gas(r) * (mass_NFW(r) + mass_gas(r))/r**2
    p = quad(dpdr, r, np.inf)[0]
    u = (3/2) * p / density_gas(r)
    return u

def sigma_2_halo(r):
    def dpdr(r):
        return 43000 * NFW(r) * (mass_NFW(r) + mass_gas(r))/r**2
    p = quad(dpdr, r, np.inf)[0]
    sigma_2 = p / NFW(r)
    return sigma_2
        
# this is just for printing out the progress in a nice way
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, 
                      fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
#############################################################################################################
######################################## CALCULATE INTEGRATION TABLES #######################################
#############################################################################################################

r_max *= r_s_DM # convert r_max into gadget units

if reuse_tables:
    tables = np.load('int_tables.npy')
    r_arr = tables[0]
    mass_arr_gas = tables[1]
    CPD_arr_gas = tables[2]
    temp_arr = tables[3]
    sigma2_arr = tables[4]

else:
    r_arr = np.logspace(-9, np.log10(r_max), N_sampling)

    print('Calculating gas mass and commulative probability distribution ... ', end='')

    mass_arr_gas = np.zeros(N_sampling)

    for i, r in enumerate(r_arr):
        mass_arr_gas[i] = mass_gas(r)
        # the CPD is proportional to the mass enclosed in r.
        # Normalize it such that CPD(r_max) = 1:
    CPD_arr_gas = mass_arr_gas/mass_arr_gas[N_sampling -1]

    print('done!')
    print('Calculating gas temperature profile... ')

    temp_arr = np.zeros(N_sampling)

    for i, r in enumerate(r_arr):
        printProgressBar(i, len(r_arr), length=50)
        temp_arr[i] = temp_gas(r)

    print('done!')
    print('Calculating DM velocity dispersion profile...')

    sigma2_arr = np.zeros(N_sampling)
    
    for i, r in enumerate(r_arr):
        printProgressBar(i, len(r_arr), length=50)
        sigma2_arr[i] = sigma_2_halo(r)

    # save tables for later use
    tables = np.array([r_arr, mass_arr_gas, CPD_arr_gas, temp_arr, sigma2_arr])
    np.save('int_tables', tables)

total_mass_gas = mass_arr_gas[N_sampling - 1]
total_mass_DM = mass_NFW(r_max)
print('done!')
#############################################################################################################
############################################# SAMPLE PARTICLES ##############################################
#############################################################################################################

print('Sampling gas ...')

coords_gas = np.zeros([N_part_gas, 3])
vels_gas = np.zeros([N_part_gas, 3])
temps_gas = np.zeros(N_part_gas)

for i in range(N_part_gas):
    printProgressBar(i, N_part_gas, length=50)
    
    # sample radius
    x = uniform(0, 1)
    idx = find_closest(CPD_arr_gas, x)
    r = r_arr[idx]
    
    # sample angular position 
    phi = uniform(0, 1) * 2 * np.pi
    x = uniform(0.0,1.0)-0.5
    theta = np.arccos(-2.0*x)
    
    # set coordinates
    coords_gas[i][0] = r*np.sin(theta)*np.cos(phi)
    coords_gas[i][1] = r*np.sin(theta)*np.sin(phi)
    coords_gas[i][2] = r*np.cos(theta)
    
    # set temperature
    temps_gas[i] = temp_arr[idx]
    
print('')
print('Sampling DM ...')

coords_DM = np.zeros([N_part_DM, 3])
vels_DM = np.zeros([N_part_DM, 3])

for i in range(N_part_DM):
    printProgressBar(i, N_part_DM, length=50)

    # sample radius 
    x = uniform(0, 1)
    r = inverse_CPD_NFW(x)

    # sample angular position 
    phi = uniform(0, 1) * 2 * np.pi
    x = uniform(0.0,1.0)-0.5
    theta = np.arccos(-2.0*x)

    # find coordinates
    coords_DM[i][0] = r*np.sin(theta)*np.cos(phi)
    coords_DM[i][1] = r*np.sin(theta)*np.sin(phi)
    coords_DM[i][2] = r*np.cos(theta)

    # find square velocity dispersion
    idx = find_closest(r_arr, r)
    sigma2 = sigma2_arr[idx]
    sigma = np.sqrt(sigma2)

    vels_DM[i][0] = gauss(0, sigma)
    vels_DM[i][1] = gauss(0, sigma)
    vels_DM[i][2] = gauss(0, sigma)
    
#############################################################################################################
############################################# SAVE TO HDF5 FILE #############################################
#############################################################################################################

print(' ')
print('Saving ...',end='')

data = []

data_gas = {}
data_gas['count'] = N_part_gas
data_gas['PartMass'] = total_mass_gas/N_part_gas
data_gas['PartType'] = 0
data_gas['Coordinates'] = coords_gas
data_gas['Velocities'] = vels_gas
data_gas['InternalEnergy'] = temps_gas
data.append(data_gas)
    
data_DM = {}
data_DM['count'] = N_part_DM
data_DM['PartMass'] = total_mass_DM/N_part_DM
data_DM['PartType'] = 1
data_DM['Coordinates'] = coords_DM
data_DM['Velocities'] = vels_DM
data.append(data_DM)

tp.write_IC_hdf5('Cluster_DM_gas_v2', data)

print('done!')