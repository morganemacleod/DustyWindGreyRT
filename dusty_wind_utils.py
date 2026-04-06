import numpy as np
from astropy.io import ascii
import athena_read as ar

def read_trackfile(fn,m1=0,m2=0):
    orb=ascii.read(fn)
    print( "reading orbit file for dusty wind simulation...")
    
    if 'x2' in orb.colnames:
        orb['x'] = orb['x2']
        orb['y'] = orb['y2']
        orb['z'] = orb['z2']
        orb['vx'] = orb['vx2']
        orb['vy'] = orb['vy2']
        orb['vz'] = orb['vz2']
    
    orb['sep'] = np.sqrt(orb['x']**2 + orb['y']**2 + orb['z']**2)

    return orb




def read_and_rotate_data_for_rt(fn,orb,rsoft2=0.1,level=0,
                                get_cartesian=True,get_cartesian_vel=True,
                                x1_min=None,x1_max=None,
                                x2_min=None,x2_max=None,
                                x3_min=None,x3_max=None,
                                gamma=5./3.,
                                pole_dir=2,
                                dens_pres_scale_factor=1.0,
                                azim_angle=0.0,pol_angle=0.0
                                ):
    """ Read spherical data and reconstruct cartesian mesh for analysis/plotting """
    
    print("read_data...reading file",fn)
    
    
    d = ar.athdf(fn,level=level,subsample=True,
                 x1_min=x1_min,x1_max=x1_max,
                 x2_min=x2_min,x2_max=x2_max,
                 x3_min=x3_min,x3_max=x3_max,
                 quantities=['rho','press','vel1','vel2','vel3']) # approximate arrays by subsampling if level < max
    print(" ...file read, constructing arrays")
    print(" ...gamma=",gamma)

    # rotate in azim_angle
    if azim_angle < 0:
        print("ERROR: azim angle should be positive!!")
        exit()
    d['x3v'] = d['x3v']-azim_angle  #np.where(d['x3v']-azim_angle<0, d['x3v']-azim_angle + 2*np.pi, d['x3v']-azim_angle)

    # rotate in pol_angle
    if pol_angle != 0:
        print("ERROR: pol_angle other than zero not implimented yet!")
        exit()
    
    # SCALE DENSITY AND PRESSURE
    d['rho'] = dens_pres_scale_factor*d['rho']
    d['press'] = dens_pres_scale_factor*d['press']


    d['gx1v']=np.broadcast_to(d['x1v'],(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    d['gx2v']=np.swapaxes(np.broadcast_to(d['x2v'],(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    d['gx3v']=np.swapaxes(np.broadcast_to(d['x3v'],(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )

    # AREA / VOLUME 
    
    ### 
    # CARTESIAN VALUES
    ###
    if(get_cartesian or get_torque or get_energy):
        print("...getting cartesian arrays...")
        # angles
        sin_th = np.sin(d['gx2v'])
        cos_th = np.cos(d['gx2v'])
        sin_ph = np.sin(d['gx3v'])
        cos_ph = np.cos(d['gx3v'])
        del d['gx2v'],d['gx3v']
                
        # cartesian coordinates
        if(pole_dir==2):
            d['x'] = d['gx1v'] * sin_th * cos_ph 
            d['y'] = d['gx1v'] * sin_th * sin_ph 
            d['z'] = d['gx1v'] * cos_th
      
                
        del d['vel1'],d['vel2'],d['vel3']    
        del cos_th, sin_th, cos_ph, sin_ph


    ## FILL IN:
    d['kappa'] = 1e-3*np.ones_like(d['rho'])
    
    return d





def get_midplane_theta(myfile,level=0):
    dblank=ar.athdf(myfile,level=level,quantities=[],subsample=True)

    # get closest to midplane value
    return dblank['x2v'][ np.argmin(np.abs(dblank['x2v']-np.pi/2.) ) ]


def get_plot_array_midplane(arr):
    return np.append(arr,[arr[0]],axis=0)


def rcom_vcom(orb,t):
    """pass a pm_trackfile.dat that has been read, time t"""
    rcom =  np.array([np.interp(t,orb['time'],orb['rcom'][:,0]),
                  np.interp(t,orb['time'],orb['rcom'][:,1]),
                  np.interp(t,orb['time'],orb['rcom'][:,2])])
    vcom =  np.array([np.interp(t,orb['time'],orb['vcom'][:,0]),
                  np.interp(t,orb['time'],orb['vcom'][:,1]),
                  np.interp(t,orb['time'],orb['vcom'][:,2])])
    
    return rcom,vcom

def pos_secondary(orb,t):
    x2 = np.interp(t,orb['time'],orb['x'])
    y2 = np.interp(t,orb['time'],orb['y'])
    z2 = np.interp(t,orb['time'],orb['z'])
    return x2,y2,z2


### Ray tracing
from scipy.interpolate import RegularGridInterpolator

def get_interp_function(d,var):
    dph = np.gradient(d['x3v'])[0]
    x3v = np.append(d['x3v'][0]-dph,d['x3v'])
    x3v = np.append(x3v,x3v[-1]+dph)
    
    var_data = np.append([d[var][-1]],d[var],axis=0)
    var_data = np.append(var_data,[var_data[0]],axis=0)

    # return error if out of bounds so that it fails if incorrect data range is accessed
    var_interp = RegularGridInterpolator((x3v,d['x2v'],d['x1v']),var_data,bounds_error=True,method='nearest')
    return var_interp



def cart_to_polar_shifted(x,y,z,azim_angle,pol_angle):
    """ returns phi in range 0-2pi"""
    r = np.sqrt(x**2 + y**2 +z**2)
    th = np.arccos(z/r)
    phi=np.where(np.arctan2(y,x)<-azim_angle,np.arctan2(y,x) + 2.0*np.pi, np.arctan2(y,x) )
    return phi,th,r

def polar_to_cart(ph,th,r):
    x = r * np.sin(th) * np.cos(ph) 
    y = r * np.sin(th) * np.sin(ph) 
    z = r * np.cos(th)
    return x,y,z


def get_ray(ydart, zdart, rstar, inner_lim, outer_lim, N_raypoints,azim_angle,pol_angle):

    xdart = np.sqrt(1.0 - ydart**2 - zdart**2)
    origin = np.array([xdart, ydart, zdart])*rstar
    

    # ray points along +x direction
    ray={}

    points = np.linspace(xdart,(outer_lim-inner_lim+xdart),N_raypoints)
    ray['dl'] = points[1:]-points[0:-1]
    ray['l'] = 0.5*(points[1:]+points[0:-1])
    ray['x'] = origin[0] + ray['l']  
    ray['y'] = origin[1] + np.zeros_like(ray['l']) 
    ray['z'] = origin[2] + np.zeros_like(ray['l']) 
    #print("ray faces = ",points)
   
    # spherical polar
    ray['phi'],ray['theta'],ray['r'] = cart_to_polar_shifted(ray['x'],ray['y'],ray['z'],azim_angle,pol_angle)
    #print (" ... ray:\n","x:",ray['x'],"\ny:",ray['y'],"\nz:",ray['z'],"\nr:",ray['r'] )

    #print(" ... ray has l=",ray['l'][0],ray['l'][-1])
    #print(" ... ray has ",len(ray['r']),"points, between r=",ray['r'][0],ray['r'][-1], " th/pi=",ray['theta'][0]/np.pi,ray['theta'][-1]/np.pi, " phi/pi=",ray['phi'][0]/np.pi,ray['phi'][-1]/np.pi )
    
    return ray
