import numpy as np
import scipy as sp
import sys
from scipy import misc, ndimage
import matplotlib.pyplot as plt

image_name = 'image.png' # default image
if len(sys.argv) > 1:
    image_name = sys.argv[1]

im = sp.misc.imread(image_name,mode='L')
im = im.astype('int32')

Gx = ndimage.sobel(im, 0)
Gy = ndimage.sobel(im, 1)

G = np.sqrt(Gx*Gx + Gy*Gy) # gradient

fig = plt.figure()
imgplot = plt.imshow(G, cmap=plt.cm.gray)
fig.canvas.set_window_title("Edge detection on " + image_name)

plt.show() # displays the plot
