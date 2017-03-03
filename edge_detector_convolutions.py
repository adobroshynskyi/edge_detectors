import numpy as np
import sys
from scipy import signal, misc, ndimage
import matplotlib.pyplot as plt

def generate_convolutions(sigmas,thetas):
    convolutions = []
    # sigmas is [1,3,6]
    # thetas is [0,pi/2,pi/4,3pi/4]
    for sigma in sigmas:
        for theta in thetas:
            im_filter = kernel(sigma, theta) # make a filter
            convolution = signal.convolve2d(image, im_filter, mode='same') # doing the convolution
            convolutions.append(convolution)

    return convolutions

# computes max convolutions for real and complex parts
def get_maxs(convolutions):
    max_real = np.real(convolutions[0])
    max_imag = np.imag(convolutions[0])
    for convolution in convolutions:
        max_real = np.maximum(np.absolute(max_real), np.absolute(np.real(convolution)))
        max_imag = np.maximum(np.absolute(max_imag), np.absolute(np.imag(convolution)))

    return max_real, max_imag

def plot_maxs(max_real, max_imag):
    fig = plt.figure()
    fig.canvas.set_window_title("Edge detection on " + image_name)

    a=fig.add_subplot(1,2,1)
    a.set_title('convolution real part max')
    plt.imshow(max_real, cmap=plt.cm.gray)

    a=fig.add_subplot(1,2,2)
    a.set_title('convolution complex part max')
    plt.imshow(max_imag, cmap=plt.cm.gray)

def kernel(sigma, theta):
    x = np.linspace(-37/2.0, 37/2.0)
    y = np.linspace(-37/2.0, 37/2.0)
    [X, Y] = np.meshgrid(x, y)
    ue = X * np.cos(theta) + Y * np.sin(theta)
    uSq = np.power(X, 2.0) + np.power(Y, 2.0)

    b = np.exp(1j * (np.pi / (2.0 * sigma)) * ue)
    d = np.exp(-1.0 * uSq / ( 2.0 * np.power(sigma, 2.0)) )

    c2 = np.sum(b * d) / np.sum(d)
    c1 = 1.0 / np.sqrt(np.sum( 1.0 - 2.0*c2 * np.cos( ue * np.pi / (2.0*sigma)) + np.power(c2, 2.0) * np.exp(-1.0 * uSq / ( 1.0 * np.power(sigma, 2.0)) )) )

    out = c1/sigma * (b-c2) * d
    return np.transpose(out)

image_name = 'image.png' # default image
if len(sys.argv) > 1:
    image_name = sys.argv[1]

image = misc.imread(image_name,mode='L')

sigmas = [1,3,6]
thetas = [0.0, np.pi/2.0, np.pi / 4.0, (3.0*np.pi)/4.0]
convolutions = generate_convolutions(sigmas,thetas)

print "finished convolutions"

maxs = get_maxs(convolutions)

max_real = maxs[0]
max_imag = maxs[1]

plot_maxs(max_real,max_imag)

epsilon = 1000.0

# ratio
ratio = (max_real + 0.001 * epsilon) / (max_imag + epsilon)
D = ratio.max()
edge = 1.0 - (ratio / D * 1.0)

edge_plot = plt.figure()
imgplot = plt.imshow(edge, cmap=plt.cm.gray)
edge_plot.canvas.set_window_title("Edge detection on " + image_name)

plt.show() # displays the plot
