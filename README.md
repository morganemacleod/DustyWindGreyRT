# DustyWindGreyRT
post-processing for grey radiative transfer through athena++ calculation 

Example command:
<code>
python ../DustyWindGreyRT/RT.py --base_dir ./ --snapshot_filename binary_wind.prim.00053.athdf --azim_angle 2.0 --N_diameter 100 --N_raypoints 100
</code>
We can visualize images like:
<code>
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

rt = Table.read("binary_wind.prim.00053_rt.dat",format='ascii')
plt.scatter(rt['ystar'],rt['zstar'],c=rt['stellar_intensity_attenuated'],cmap='plasma',vmin=0)
plt.axis('equal')
plt.colorbar(label='relative intensity')
</code>
and we can plot the light curve as:
<code>
lc = Table.read("binary_wind.prim_lightcurve.dat",format='ascii')
plt.plot(lc['times'],lc['flux'],'d-')
</code>