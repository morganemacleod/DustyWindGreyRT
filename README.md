# DustyWindGreyRT
post-processing for grey radiative transfer through athena++ calculation 

Example command:

<code>
python ../DustyWindGreyRT/RT.py --base_dir ./ --snapshot "binary_wind.prim.00053.athdf" --scale 1.0 --N_diameter 100 --N_raypoints 100 --N_angles 10
</code>

We can visualize images like:

<code>
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

rt = Table.read("binary_wind.prim.00053_a0.698.dat",format='ascii')
plt.scatter(rt['ystar'],rt['zstar'],c=rt['stellar_intensity_attenuated'])
plt.axis('equal')
plt.colorbar()
</code>

and we can plot the light curve as (note double check sign on angle/time):
<code>
lc = Table.read("binary_wind.prim.00053_lightcurve.dat",format='ascii')
plt.plot(lc['angle']/2*np.pi,lc['flux'])
</code>