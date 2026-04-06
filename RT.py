import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import Constants
import dusty_wind_utils as dw
from astropy.table import Table
from glob import glob

c=Constants.Constants()

# Quadratic LD coeff V band http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/554/A98#/browse
# http://cdsarc.u-strasbg.fr/ftp/J/A+A/554/A98/ReadMe
# USING TABLE 3,
#   3600  -0.25   10.0   1.4631  -0.3821   1.3159  -0.2080   1.1055   0.0242   0.7179   0.4584   0.3054   0.9619   0.0813   1.2058   0.9838   0.1613   1.0586   0.0765
ld1 = 1.3159
ld2 = -0.2080


def I(mu, ld1, ld2):
    return np.where(mu == 0.0, 0.0, (1. - ld1 * (1. - mu) - ld2 * (1. - mu) ** 2))


def generate_uniform(N_diameter):
    y = np.linspace(-1,1,N_diameter)
    z = np.linspace(-1,1,N_diameter)
    yy,zz = np.meshgrid(y,z)
    y_circle = yy[yy**2 + zz**2 < 1].copy()
    z_circle = zz[yy**2 + zz**2 < 1].copy()
    return y_circle, z_circle
    


def MC_ray(dart):
    """ computes sum of tau along LOS of a ray defined by integer 'dart' """
    ydart = yrandom[dart]
    zdart = zrandom[dart]
    
    ray = dw.get_ray(ydart=ydart,
                     zdart=zdart,
                     rstar=rad_star,
                     inner_lim=in_lim,
                     outer_lim=out_lim,
                     N_raypoints=N_raypoints,
                     azim_angle=azim_angle,
                     pol_angle=pol_angle)

    ray['rho']   =  rho_interp((ray['phi'], ray['theta'], ray['r']))
    ray['kappa'] =  kappa_interp((ray['phi'], ray['theta'], ray['r']))
    
    tauLOS = np.sum(ray['rho']*ray['kappa']*ray['dl'])
    expfac = np.exp(-tauLOS)

    print('dart (i,y/R_star,z/R_star, az_angle, tau, e(-tau)): ', dart, np.round(ydart,3) , np.round(zdart,3) , np.round(azim_angle,3), np.round(tauLOS,4), np.round(expfac,4) )

    
    return expfac


parser = argparse.ArgumentParser(
    description='Read input/output directories, MC ray properties, example usage: "python BG_RT.py --base_dir ~/Dropbox/PlanetWind/Analysis/testdata/ --snapshot PW_W107.out1.00100.athdf --level 1 --N_mc 100 --N_raypoints 200 --angles 0" ')

parser.add_argument("--base_dir", type=str,help="data directory (should end with / )")
parser.add_argument("--snapshot_filename", type=str,help="example filename of snapshot to be processed")
parser.add_argument("--snapshot_start", type=int,default=0,help="snapshot number to start processing")
parser.add_argument("--snapshot_end", type=int,default=-1,help="snapshot number to end processing")
parser.add_argument("--snapshot_skip", type=int,default=1,help="process every __ snapshot")
parser.add_argument("--level", default=None, type=int,
                    help="refinement level to read the snapshot at")
parser.add_argument("--N_diameter", default=10, type=int, help="Number of linear spaced rays across diameter")
parser.add_argument("--N_raypoints", default=10, type=int,
                    help="controls num of points along a ray")
parser.add_argument("--scale", default=1.0, type=float, help="scale density and pressure by this factor")
parser.add_argument("--pol_angle", default=0.0, type=float, help='angle to rotate the snapshot from equator radians, +/- pi/2')
parser.add_argument("--azim_angle",default=0.0,type=float,help='angle to view the snapshot from, radian, 0-2pi')

args = parser.parse_args()
base_dir = args.base_dir
snapshot_filename = args.snapshot_filename
snapshot_start = args.snapshot_start
snapshot_end   = args.snapshot_end
snapshot_skip  = args.snapshot_skip
mylevel = args.level
dens_pres_scale = args.scale
N_diameter = args.N_diameter
N_raypoints = args.N_raypoints +1 
azim_angle=args.azim_angle
pol_angle =args.pol_angle

################################################################

print("STARTING RT:\n args:\n",args)




print("processing RT from (azim,pol) viewing angle:",(azim_angle,pol_angle))

    
orb = dw.read_trackfile(base_dir + "pm_trackfile.dat")
a = orb['sep'][0]


# NOTE: needs to be a full 3D output, not a slice!!!
myfile = base_dir + snapshot_filename
gamma = 5/3.

filelist = sorted(glob(myfile[0:-11]+'*'+myfile[-6:]))
print("processing the following filelist:\n",filelist)


### START LOOP OVER FILES
times =  np.zeros_like(filelist)
fluxes = np.zeros_like(filelist)
for i,fn in enumerate(filelist):

    d = dw.read_and_rotate_data_for_rt(myfile, orb, level=mylevel,gamma=gamma,dens_pres_scale_factor=dens_pres_scale,
                                       azim_angle=azim_angle,pol_angle=pol_angle)

    # NOTE: SET THESE IF DON'T WANT FULL DOMAIN (could do from 1.5 rstar to 5 rstar, for example)
    rad_star = d['x1v'][0]
    out_lim =  d['x1v'][-1]
    in_lim = rad_star
    print("computing rays between r (in,out) = ", in_lim, out_lim)

    t = d['Time'] 

    # Get interpolating functions 
    rho_interp = dw.get_interp_function(d, "rho")
    kappa_interp = dw.get_interp_function(d, "kappa")

    # Ray Tracing
    # get ray positions
    yrandom, zrandom = generate_uniform(N_diameter)
    weights = np.ones_like(yrandom)
    N_mc = len(yrandom)
    
    # calculate stellar intensity profile
    r_prime_mag = np.sqrt(yrandom * yrandom + zrandom * zrandom)
    m = np.sqrt(1. - r_prime_mag ** 2)
    stellar_intensity = weights * I(m, ld1, ld2)  # Apply the weights!
    total_stellar_intensity = np.sum(stellar_intensity)

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
    ray_table.write(fn[0:-6] + "_rt.dat", format = 'ascii',overwrite=True)
    
    
    final_intensity = total / N_mc
    final_control = control / N_mc
    print("N_mc= ", N_mc, final_intensity/final_control)
    times[i] = t
    fluxes[i] = final_intensity/final_control

        
### END LOOP OVER FILES
    
## save the angle, fluxes arrays as astropy Table
lc_flux_table = Table([times,fluxes],names=['times','flux']) 
lc_flux_table.write(fn[0:-12] + "_lightcurve.dat", format = 'ascii',overwrite=True)
print(lc_flux_table)
