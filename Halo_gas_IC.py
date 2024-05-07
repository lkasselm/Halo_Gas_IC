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
Parallel = False # Use multiple threads 

have_gas = True
have_DM = True

#gas_profile = 'Burkert'
gas_profile = 'Hernquist'

filename = 'dSph_normal_DM1e6_gas1e5'

N_sampling = 10000 # size of the arrays created in numerical integration

N_part_gas = int(1e5) # number of particles to sample
r_s_gas = 0.4 # normal dsph
#r_s_gas = 194.78 # cluster
rho_s_gas = 0.26 # normal dsph
# rho_s_gas = 9.72e-5 # cluster

N_part_DM = int(1e6)
r_max = 20 # maximum sampling distance of the system in units of r_s_DM
r_s_DM = 9.5 # dSph
rho_s_DM = 4.022e-4 # dSph

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
    x = r/r_s_gas
    return rho_s_gas/(x * (1+x)**3)

def mass_Hernquist(r):
    # return mass enclosed in r
    x = r/r_s_gas
    return 2 * np.pi * rho_s_gas * r_s_gas**3 * x**2/(x+1)**2

def inverse_CPD_Hernquist(x):
    x *= (r_max_gas/r_s_gas)**2/( (r_max_gas/r_s_gas) +1)**2
    return - r_s_gas * np.sqrt(x)/(np.sqrt(x)-1)

def mass_NFW(r):
    # return mass enclosed in r
    x = r/r_s_DM
    return 2 * np.pi * rho_s_DM * r_s_DM**3 * np.log(x**2 + 1)

def inverse_CPD_NFW(x):
    # draw random radius from inverse CPD for NFW profile.
    # assumes that x is drawn from a uniform distribution 
    # between 0 and 1. 
    # r_max is the maximum sampling distance 
    # (has to be finite in this case)
    x *= np.log((r_max_DM/r_s_DM)**2 +1)
    return r_s_DM * np.sqrt(np.exp(x) -1)

def mass_Burkert(r):
    def dMdr(r):
        return 4 * np.pi * Burkert(r) * r**2
    return quad(dMdr, 0, r)[0]

if gas_profile == 'Burkert':
    def mass_gas(r):
        return mass_Burkert(r)
    def density_gas(r):
        return Burkert(r)
    
elif gas_profile == 'Hernquist':
    def mass_gas(r):    
        return mass_Hernquist(r)
    def density_gas(r):
        return Hernquist(r)

def mean_particle_separation_gas(r):
    N = mass_gas(r)/part_mass_gas # number of particles enclosed in r
    n = 3*N/(4 * np.pi * r**3) # average number density
    return n**(-1/3) # mean seperation

def mean_particle_separation_DM(r):
    N = mass_NFW(r)/part_mass_DM # number of particles enclosed in r
    n = 3*N/(4 * np.pi * r**3) # average number density
    return n**(-1/3) # mean seperation
    
def temp_gas(r):
    def dpdr(r):
        return  43000 * density_gas(r) * total_mass(r)/r**2
    p = quad(dpdr, r, np.inf)[0]
    u = (3/2) * p / density_gas(r)
    return u

def total_mass(r):
    return have_DM * mass_NFW(r) + have_gas * mass_gas(r)

def sigma_2_halo(r):
    def dpdr(r):
        return 43000 * NFW(r) * total_mass(r)/r**2
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

r_max_gas = r_max * r_s_gas
r_max_DM = r_max * r_s_DM

if reuse_tables:
    tables = np.load('int_tables.npy')
    r_arr_gas = tables[0]
    r_arr_DM = tables[1]
    mass_arr_gas = tables[2]
    CPD_arr_gas = tables[3]
    temp_arr = tables[4]
    sigma2_arr = tables[5]

else:
    r_arr_gas = np.logspace(-9+np.log10(r_max_gas), np.log10(r_max_gas), N_sampling)
    r_arr_DM = np.logspace(-9+np.log10(r_max_DM), np.log10(r_max_DM), N_sampling)
    
    if have_gas:
        print('Calculating gas temperature profile... ')

        temp_arr = np.zeros(N_sampling)

        for i, r in enumerate(r_arr_gas):
            printProgressBar(i, len(r_arr_gas), length=50)
            temp_arr[i] = temp_gas(r)

    if have_DM:
        print('')
        print('Calculating DM velocity dispersion profile...')

        sigma2_arr = np.zeros(N_sampling)

        for i, r in enumerate(r_arr_DM):
            printProgressBar(i, len(r_arr_DM), length=50)
            sigma2_arr[i] = sigma_2_halo(r)

        # save tables for later use
        tables = np.array([r_arr_gas, r_arr_DM, temp_arr, sigma2_arr])
        np.save('int_tables', tables)
    
total_mass_gas = mass_gas(r_max_gas)
total_mass_DM = mass_NFW(r_max_DM)

#############################################################################################################
############################################# SAMPLE PARTICLES ##############################################
#############################################################################################################

if have_gas:
    
    print('')
    print('Sampling gas ...')

    coords_gas = np.zeros([N_part_gas, 3])
    vels_gas = np.zeros([N_part_gas, 3])
    temps_gas = np.zeros(N_part_gas)

    for i in range(N_part_gas):
        printProgressBar(i, N_part_gas, length=50)

        # sample radius
        x = uniform(0, 1)
        r = inverse_CPD_Hernquist(x)

        # sample angular position 
        phi = uniform(0, 1) * 2 * np.pi
        x = uniform(0.0,1.0)-0.5
        theta = np.arccos(-2.0*x)

        # set coordinates
        coords_gas[i][0] = r*np.sin(theta)*np.cos(phi)
        coords_gas[i][1] = r*np.sin(theta)*np.sin(phi)
        coords_gas[i][2] = r*np.cos(theta)

        # set temperature
        idx = find_closest(r_arr_gas, r)
        temps_gas[i] = temp_arr[idx]
    
if have_DM:
    print('')
    print('Sampling DM ...')

    coords_DM = np.zeros([N_part_DM,3])
    vels_DM = np.zeros([N_part_DM,3])

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
        idx = find_closest(r_arr_DM, r)
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

if have_gas:
    part_mass_gas = total_mass_gas/N_part_gas
    data_gas = {}
    data_gas['count'] = N_part_gas
    data_gas['PartMass'] = part_mass_gas
    data_gas['PartType'] = 0
    data_gas['Coordinates'] = coords_gas
    data_gas['Velocities'] = vels_gas
    data_gas['InternalEnergy'] = temps_gas
    data.append(data_gas)

if have_DM:
    part_mass_DM = total_mass_DM/N_part_DM
    data_DM = {}
    data_DM['count'] = N_part_DM
    data_DM['PartMass'] = part_mass_DM
    data_DM['PartType'] = 1
    data_DM['Coordinates'] = coords_DM
    data_DM['Velocities'] = vels_DM
    data.append(data_DM)

tp.write_IC_hdf5(filename, data)

print('done!')
print('Suggested softening lengths based on mean central interparticle spacing:')
if have_gas:
    print('Gas: ', np.round(mean_particle_separation_gas(r_s_gas/5)*2, 3), ' kpc')
if have_DM:
    print('DM: ', np.round(mean_particle_separation_DM(r_s_gas/5)*2, 3), ' kpc')
