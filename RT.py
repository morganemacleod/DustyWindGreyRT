import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import Constants
import dusty_wind_utils as dw
from astropy.table import Table

c=Constants.Constants()

# Quadratic LD coeff V band http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/554/A98#/browse
# http://cdsarc.u-strasbg.fr/ftp/J/A+A/554/A98/ReadMe
# USING TABLE 3,
#   3600  -0.25   10.0   1.4631  -0.3821   1.3159  -0.2080   1.1055   0.0242   0.7179   0.4584   0.3054   0.9619   0.0813   1.2058   0.9838   0.1613   1.0586   0.0765
ld1 = 1.3159
ld2 = -0.2080


def I(mu, ld1, ld2):
    return np.where(mu == 0.0, 0.0, (1. - ld1 * (1. - mu) - ld2 * (1. - mu) ** 2))

def New_get_interp_function(d, var):
    dph = np.gradient(d['x3v'])[0]
    x3v = np.append(d['x3v'][0] - dph, d['x3v'])
    x3v = np.append(x3v, x3v[-1] + dph)

    var_data = np.append([var[-1]], var, axis=0)
    var_data = np.append(var_data, [var_data[0]], axis=0)

    var_interp = RegularGridInterpolator(
        (x3v, d['x2v'], d['x1v']), var_data, bounds_error=True)
    return var_interp

def generate_uniform(N_diameter):
    y = np.linspace(-1,1,N_diameter)
    z = np.linspace(-1,1,N_diameter)
    yy,zz = np.meshgrid(y,z)
    y_circle = yy[yy**2 + zz**2 < 1].copy()
    z_circle = zz[yy**2 + zz**2 < 1].copy()
    return y_circle, z_circle
    

def generate_random(N_mc):
    theta = 2 * np.pi * np.random.random_sample(N_mc)
    r = np.sqrt(np.random.random_sample(N_mc))

    yrandom = r * np.cos(theta)
    zrandom = r * np.sin(theta)

    return yrandom, zrandom



def MC_ray(dart):
    """ computes sum of tau along LOS of a ray defined by integer 'dart' """
    ydart = yrandom[dart]
    zdart = zrandom[dart]
    
    ray = dw.get_ray(planet_pos=(x2, y2, z2),
                     ydart=ydart,
                     zdart=zdart,
                     azim_angle=azim_angle,
                     #pol_angle=pol_angle, 
                     rstar=rad_star,
                     inner_lim=in_lim,
                     outer_lim=out_lim,
                     N_raypoints=N_raypoints)

    ray['rho']   =  rho_interp((ray['phi'], ray['theta'], ray['r']))
    ray['kappa'] =  kappa_interp((ray['phi'], ray['theta'], ray['r']))
    
    tauLOS = np.sum(ray['rho']*ray['kappa']*ray['dl'])
    expfac = np.exp(-tauLOS)

    print('dart (i,y/R_star,z/R_star, az_angle, tau, e(-tau)): ', dart, np.round(ydart,3) , np.round(zdart,3) , np.round(azim_angle,3), np.round(tauLOS,4), np.round(expfac,4) )

    
    return expfac


parser = argparse.ArgumentParser(
    description='Read input/output directories, MC ray properties, example usage: "python BG_RT.py --base_dir ~/Dropbox/PlanetWind/Analysis/testdata/ --snapshot PW_W107.out1.00100.athdf --level 1 --N_mc 100 --N_raypoints 200 --angles 0" ')

parser.add_argument("--base_dir", type=str,help="data directory (should end with / )")
parser.add_argument("--snapshot", type=str,help="filename of snapshot to be processed")
parser.add_argument("--N_angles", type=int,
                    help="angles at which to perform the RT", required=True)
parser.add_argument("--level", default=None, type=int,
                    help="refinement level to read the snapshot at")
parser.add_argument("--N_diameter", default=10, type=int, help="Number of linear spaced rays across diameter")
parser.add_argument("--N_raypoints", default=10, type=int,
                    help="controls num of points along a ray")
parser.add_argument("--scale", default=1.0, type=float, help="scale density and pressure by this factor")
#parser.add_argument("--bplanet", default=0.0, type=float, help='impact parameter in stellar radii')

args = parser.parse_args()
base_dir = args.base_dir
snapshot = args.snapshot
mylevel = args.level
# N_mc = args.N_mc
# nraypoints = args.N_raypoints
angles = np.linspace(0,2*np.pi,args.N_angles)
dens_pres_scale = args.scale
N_diameter = args.N_diameter
N_raypoints = args.N_raypoints 
#bplanet = args.bplanet


################################################################


orb = dw.read_trackfile(base_dir + "pm_trackfile.dat")
a = orb['sep'][0]


# NOTE: needs to be a full 3D output, not a slice!!!
myfile = base_dir + snapshot
gamma = 5/3.

d = dw.read_data_for_rt(myfile, orb, level=mylevel,gamma=gamma,dens_pres_scale_factor=dens_pres_scale)

# NOTE: SET THESE IF DON'T WANT FULL DOMAIN (could do from 1.5 rstar to 5 rstar, for example)
rad_star = d['x1v'][0]
out_lim =  d['x1v'][-1]
in_lim = rad_star
print("computing rays between r (in,out) = ", in_lim, out_lim)

# rotation angle to achieve impact parameter b 
#pol_angle = - np.arcsin(bplanet*rad_star / a)
#print("impact parameter b=",bplanet,"rotating by ",pol_angle) 


t = d['Time']
x2, y2, z2 = dw.pos_secondary(orb, t)
print('Time:', t)
print('Position of secondary: ', x2, y2, z2)


d2 = np.sqrt((d['x'] - x2) ** 2 + (d['y'] - y2) ** 2 + (d['z'] - z2) ** 2)

dr = np.broadcast_to(d['x1f'][1:] - d['x1f'][0:-1],
                     (len(d['x3v']), len(d['x2v']), len(d['x1v'])))

################################################################
# Get interpolating functions 
rho_interp = dw.get_interp_function(d, "rho")
kappa_interp = dw.get_interp_function(d, "kappa")

###############################################################

# ray tracing
print("ray tracing for", len(angles), "angles =", angles)
fluxes = np.zeros_like(angles)
aind = 0

for i,aa in enumerate(angles):
    print("#####\n angle=", aa, "######")
    aind += 1
    azim_angle = aa

    # get ray positions
    yrandom, zrandom = generate_uniform(N_diameter)
    weights = np.ones_like(yrandom)
    N_mc = len(yrandom)
    
    # calculate stellar intensity profile
    r_prime_mag = np.sqrt(yrandom * yrandom + zrandom * zrandom)
    m = np.sqrt(1. - r_prime_mag ** 2)
    stellar_intensity = weights * I(m, ld1, ld2)  # Apply the weights!
    total_stellar_intensity = np.sum(stellar_intensity)
    # print "total_stellar_intensity=",total_stellar_intensity



    ### sum, checking for nan
    total = 0.0
    control = 0.0

    stellar_intensity_attenuated = np.zeros_like(stellar_intensity)
    for dart in range(N_mc):
        # compute the RT
        exp_fac = MC_ray(dart)
        stellar_intensity_attenuated[dart] = stellar_intensity[dart] * exp_fac

        if np.isnan(exp_fac):
            print('nan dart !!!!! ')
        else:
            total    += stellar_intensity[dart] * exp_fac
            control  += stellar_intensity[dart]


    # Make a table of rays, save for making images of star
    ray_table = Table( [yrandom,zrandom,weights,stellar_intensity,stellar_intensity_attenuated], names=['ystar', 'zstar','weight','stellar_intensity','stellar_intensity_attenuated'] )
    ray_table.write(base_dir + snapshot[0:-6] + "_a" + str(np.round(aa,3))+ ".dat", format = 'ascii',overwrite=True)
    
    
    final_intensity = total / N_mc
    final_control = control / N_mc
    print("N_mc= ", N_mc, final_intensity/final_control)
    fluxes[i] = final_intensity/final_control


    
## save the angle, fluxes arrays as astropy Table
angle_flux_table = Table([angles,fluxes],names=['angle','flux']) 
angle_flux_table.write(base_dir + snapshot[0:-6] + "_lightcurve.dat", format = 'ascii',overwrite=True)
print(angle_flux_table)
