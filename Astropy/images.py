import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

# # Read in file
# hdulist = fits.open('/Users/onorinbejasus/Downloads/photoField-002267-1.fits')
# print hdulist
# 
# # Extract image and header
# image = hdulist[0].data
# header = hdulist[0].header
# 
# # Instantiate WCS object
# w = WCS(header)
# 
# # Plot the image
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.imshow(image, origin='lower')
# 
# # Loop over lines of longitude
# for lon in np.linspace(-180., 180., 13):
#     grid_lon = np.repeat(lon, 100)
#     grid_lat = np.linspace(-90., 90., 100)
#     px, py = w.wcs_world2pix(grid_lon, grid_lat, 1)
#     ax.plot(px, py, color='yellow', alpha=0.5)
# 
# # Loop over lines of latitude
# for lat in np.linspace(-60., 60., 5):
#     grid_lon = np.linspace(-180., 180., 100)
#     grid_lat = np.repeat(lat, 100)
#     px, py = w.wcs_world2pix(grid_lon, grid_lat, 1)
#     ax.plot(px, py, color='yellow', alpha=0.5)
# 
# ra = float(header['CRPIX1'])
# xa = float(header['CRPIX2'])
# circle1=plt.Circle((ra,xa), 2,color='b')
# fig.gca().add_artist(circle1)
# 
# ax.set_xlim(0, image.shape[1])
# ax.set_ylim(0, image.shape[0])
# ax.set_xticklabels('')
# ax.set_yticklabels('')
# fig.savefig('wcs_extra.png', bbox_inches='tight')

from astropy.table import Table
from matplotlib import pyplot as plt

# a = [1, 2, 3]
# b = [4.0, 5.0, 6.2]
# c = ['x', 'y', 'z']
# t = Table([a, b, c], names=('a', 'b', 'c'))

t = Table.read('table.vot', format='votable')
t.keep_columns(['RAJ2000', 'DEJ2000', 'Count'])
print t
t_bright = t[t['Count'] > 2.]
th_bright = t[t['Count'] > 3.]

m_max = th_bright[0]
for m_t in th_bright:
    if m_t['Count'] > m_max:
        m_max = m_t['Count']

bright = m_max
print bright["Count"]

fig = plt.figure()
ax = fig.add_subplot(1,1,1, aspect='equal')
ax.scatter(t['RAJ2000'], t['DEJ2000'], s=1, color='black')
ax.scatter(t_bright['RAJ2000'], t_bright['DEJ2000'], color='red')
ax.scatter(th_bright['RAJ2000'], th_bright['DEJ2000'], color='blue')
ax.scatter(bright['RAJ2000'], bright['DEJ2000'], color='yellow')
ax.set_xlim(360., 0.)
ax.set_ylim(-90., 90.)
ax.set_xlabel("Right Ascension")
ax.set_ylabel("Declination")

fig.savefig('tables_level2.png', bbox_inches='tight')