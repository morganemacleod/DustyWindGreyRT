import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import dusty_wind_utils as dw

c=Constants()

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


def generate_random(N_mc):
    theta = 2 * np.pi * np.random.random_sample(N_mc)
    r = np.sqrt(np.random.random_sample(N_mc))

    yrandom = r * np.cos(theta)
    zrandom = r * np.sin(theta)

    return yrandom, zrandom


def generate_rays_weighted(Nr, slope, yp, zp, rad_planet_frac):
    """Nr is the number of radius bins from the planet, slope is the power law sampling (1=linear, <1 is centrally concentrated)"""
    # rf = np.logspace(np.log10(rad_planet_frac),np.log10(1+np.sqrt(yp**2 + zp**2)),Nr) # faces of the rings
    rf = np.linspace(rad_planet_frac ** slope, (1 + np.sqrt(yp ** 2 + zp ** 2)) ** slope, Nr) ** (1 / slope)
    ra = (2 / 3) * (rf[1:] ** 3 - rf[0:-1] ** 3) / (rf[1:] ** 2 - rf[0:-1] ** 2)  ## Area weighted centers
    dr = rf[1:] - rf[0:-1]

    rr = []
    tt = []
    da = []
    for i in range(len(ra)):
        Nth = int(np.round(2 * np.pi * ra[i] / dr[i]))
        th = np.linspace(0, 2 * np.pi, Nth + 1)
        th = 0.5 * (th[1:] + th[0:-1])
        dth = th[1] - th[0]
        for j in range(Nth):
            rr.append(ra[i])
            tt.append(th[j])
            da.append(np.pi * (rf[i + 1] ** 2 - rf[i] ** 2) / Nth)

    rr = np.array(rr).flatten()
    tt = np.array(tt).flatten()
    da = np.array(da).flatten()

    yrays = yp + rr * np.cos(tt)
    zrays = zp + rr * np.sin(tt)

    sel = np.sqrt(yrays ** 2 + zrays ** 2) < 1.0
    yrays = yrays[sel].copy()
    zrays = zrays[sel].copy()
    da = da[sel].copy()
    print("selected N=", len(yrays), "rays")

    return yrays, zrays, da





def MC_ray(dart):
    """ computes sum of tau along LOS of a ray defined by integer 'dart' """
    ydart = yrandom[dart]
    zdart = zrandom[dart]
    print('dart (i,y/R_star,z/R_star, az_angle, pol_angle): ', dart, np.round(ydart,3) , np.round(zdart,3) , np.round(azim_angle,3) ,np.round(pol_angle,3) )

    ray = dw.get_ray(planet_pos=(x2, y2, z2),
                     ydart=ydart,
                     zdart=zdart,
                     azim_angle=azim_angle,
                     pol_angle=pol_angle, 
                     rstar=rad_star,
                     rplanet=rp,
                     fstep=f_raystep,
                     inner_lim=in_lim,
                     outer_lim=out_lim)

    ray['rho']   =  rho_interp((ray['phi'], ray['theta'], ray['r']))
    ray['kappa'] =  kappa_interp((ray['phi'], ray['theta'], ray['r']))
    
    tauLOS = np.sum(ray['rho']*ray['kappa']*ray['dl'])
    expfac = np.exp(-tauLOS)

    return expfac


parser = argparse.ArgumentParser(
    description='Read input/output directories, MC ray properties, example usage: "python BG_RT.py --base_dir ~/Dropbox/PlanetWind/Analysis/testdata/ --snapshot PW_W107.out1.00100.athdf --level 1 --N_mc 100 --N_raypoints 200 --angles 0" ')

parser.add_argument("--base_dir", help="data directory (should end with / )")
parser.add_argument("--snapshot", help="filename of snapshot to be processed")
parser.add_argument("--angles", type=float, nargs='+',
                    help="angles at which to perform the spectral synthesis (radians, mid-transit=0)", required=True)
parser.add_argument("--level", default=1, type=int,
                    help="refinement level to read the snapshot at")
# parser.add_argument("--N_mc", default=1000, type=int, help="number of MC rays")
parser.add_argument("--N_radial", default=30, type=int, help="number of radial bins for RT rays")
parser.add_argument("--f_raystep", default=0.2, type=float,
                    help="controls num of points along a ray, dl=f_raystep*dplanet")
parser.add_argument("--scale", default=1.0, type=float, help="scale density and pressure by this factor")
parser.add_argument("--rstar", default=4.67e10, type=float,
                    help="stellar radius, usually the inner radius of spherical polar mesh x1min")
parser.add_argument("--bplanet", default=0.0, type=float, help='impact parameter in stellar radii')

args = parser.parse_args()
base_dir = args.base_dir
snapshot = args.snapshot
mylevel = args.level
# N_mc = args.N_mc
# nraypoints = args.N_raypoints
angles = args.angles
dens_pres_scale = args.scale
N_radial = args.N_radial
f_raystep = args.f_raystep
rad_star = args.rstar
bplanet = args.bplanet


################################################################


orb = dw.read_trackfile(base_dir + "pm_trackfile.dat")

################################################################

m1 = orb['m1'][0]
m2 = orb['m2'][0]
a = orb['sep'][0]

print(" running RT for: ")
print("a = ", a)
print("rstar =", rad_star)


################################################################

# NOTE: needs to be a full 3D output, not a slice!!!
myfile = base_dir + snapshot
# NOTE: SET THESE 
out_lim = 2 * a 
in_lim = 0.5 * a  
print("computing rays between r (in,out) = ", in_lim, out_lim)

gamma = 5/3.

# rotation angle to achieve impact parameter b 
pol_angle = - np.arcsin(bplanet*rad_star / a)
print("impact parameter b=",bplanet,"rotating by ",pol_angle) 

# NOTE: setting limits for reading the data. These probably could be adjusted but are meant to err on the side of capturing everything. 
x1_min = max( in_lim - rad_star, rad_star) 
x1_max = max( out_lim + rad_star, 1.1*out_lim)
inner_pol_angle = np.arcsin(rad_star/in_lim)
theta_require = 1.01*np.sin(rad_star/x1_min) / np.cos(np.abs(inner_pol_angle) + np.abs(pol_angle))
print("require an angle of", theta_require/np.pi, "pi to capture stellar radius at inner limit of grid")
x2_min = max(0.0,np.pi/2 - theta_require + pol_angle)
x2_max = min(np.pi,np.pi/2 + theta_require + pol_angle)
x3_min = max(0.0,np.pi + np.min(angles) - theta_require)
x3_max = min(2*np.pi,np.pi + np.max(angles) + theta_require)

# If there are bounds errors with rays, check the range of data being read in... 
print("reading data with limits: \n x1:",x1_min,x1_max,"\n x2/pi:",x2_min/np.pi, x2_max/np.pi,"\n x3/pi:",x3_min/np.pi, x3_max/np.pi,"\n")
d = dw.read_data_for_rt(myfile, orb, level=mylevel,
                        x3_min=x3_min, x3_max=x3_max,
                        x2_min=x2_min, x2_max=x2_max,
                        x1_min=x1_min,x1_max=x1_max,
                        gamma=gamma,dens_pres_scale_factor=dens_pres_scale)


t = d['Time']
rcom, vcom = dw.rcom_vcom(orb, t)
x2, y2, z2 = dw.pos_secondary(orb, t)
print('Time:', t)
print('Position of secondary: ', x2, y2, z2)
print('time to read file:', time.time() - start_read_time)

d2 = np.sqrt((d['x'] - x2) ** 2 + (d['y'] - y2) ** 2 + (d['z'] - z2) ** 2)

dr = np.broadcast_to(d['x1f'][1:] - d['x1f'][0:-1],
                     (len(d['x3v']), len(d['x2v']), len(d['x1v'])))


#################################################################
# Convert rotating -> Inertial frame
d['vx'] = d['vx'] - Omega_orb * d['y']
d['vy'] = d['vy'] + Omega_orb * d['x']

################################################################
# Get interpolating functions #### TO DO: FILL IN
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
    azim_angle = aa + np.pi

    # get ray positions
    # yrandom, zrandom = generate_random(N_mc)
    # weights = np.ones_like(yrandom)
    # fractional y,z position of the planet within the star for weighted sampling
    yplanet = np.sin(aa) * a / rad_star
    zplanet = bplanet
    
    # yrandom,zrandom,weights = generate_random_weighted(N_mc,yplanet,zplanet,rp/rad_star)
    yrandom, zrandom, weights = generate_rays_weighted(N_radial, 0.75, yplanet, zplanet, rp / rad_star)
    N_mc = len(yrandom)

    r_prime_mag = np.sqrt(yrandom * yrandom + zrandom * zrandom)

    # calculate stellar intensity profile
    m = np.sqrt(1. - r_prime_mag ** 2)
    stellar_intensity = weights * I(m, ld1, ld2)  # Apply the weights!
    total_stellar_intensity = np.sum(stellar_intensity)
    # print "total_stellar_intensity=",total_stellar_intensity

    total = 0.0
    control = 0.0
   
    for dart in range(N_mc):
        # compute the RT
        exp_fac = MC_ray(dart)

        if np.isnan(np.sum(exp_fac)) or np.isinf(-np.log(exp_fac[int(len(nu) / 2)])):
            print('nan dart !!!!! ')
        else:
            total    += stellar_intensity[dart] * exp_fac
            control  += stellar_intensity[dart]

    final_intensity = total / N_mc
    final_control = control / N_mc
    print("N_mc= ", N_mc, final_intensity/final_control)
    fluxes[i] = final_intensity/final_control


## TO DO
## save the angle, fluxes arrays as astropy Table
