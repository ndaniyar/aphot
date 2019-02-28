# Copyright (C) 2013 Daniyar Nurgaliev
#
# This file is a part of library for computing Photon Asymmetry (A_phot)
# parameter for morphological classification of cluster.
# This parameter is described in detail in the following publication:
#
# Please, cite it if you use A_phot in your research.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from numpy import *
from numpy.fft import rfft2, irfft2, fftshift
from scipy.integrate import romb
from collections import namedtuple


#### Numpy #########################################
def arr(*a): return array(a)
def intround(a): return around(a).astype(int)
def ind(a): return arange(len(a))
def maxxy(a) : return arr(a.argmax() % a.shape[1], a.argmax() // a.shape[1])


#### Types ########################################
Ellipse = namedtuple('Ellipse', ['xc','yc','a','b','theta'])

class sdict(dict):
    def __getattr__(self, attr):
        if attr.startswith('__'): raise AttributeError
        return self[attr]
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__
    def __iadd__(self, other):
        self.update(other)
        return self
    def __add__(self, other):
        temp = sdict()
        temp.update(self)
        temp.update(other)
        return temp

class XY(ndarray):
    ''' Adds properties "x" and "y" to a 2-element array
    '''
    def __new__(cls, *args):
        if len(args)==1: args = args[0]
        return array(args).view(cls)
    def __getattr__(self, name):
        if name=='x': return self[0]
        if name=='y': return self[1]
        raise AttributeError


#### Gaussian kernels for convolutions ####################
def gauss(sigmal, normalize=True):
    return_first = False
    if isscalar(sigmal): 
        sigmal = [sigmal]
        return_first = True

    ans = []
    for sigma in sigmal:
        s = ceil(sigma)
        y,x = mgrid[-3*s:3*s,-3*s:3*s]
        ker = exp( -(x*x+y*y)/2/sigma**2 )
        if normalize: ker /= ker.sum()
        ans += [ker]

    if return_first: return ans[0]
    else: return ans

kersize = list(exp(arange(0,3.51,0.25)))
kernels = gauss(kersize, normalize=False)
norms = array([_k.sum() for _k in kernels])
nkernels = [_k/_n for _k,_n in zip(kernels,norms)]
b = int(kernels[-1].shape[0]/2)
assert type(b)==int


#### Miscellaneous #################################
def ximage(evt, padding=False):
    b, d = evt.b, evt.d
    im = zeros((2*d,2*d))
    for x,y in zip(evt.xc, evt.yc): im[y+d, x+d] += 1
    if padding: return im
    else: return im[b:-b,b:-b]


def conv(im, ker):
    ''' Convolves image im with kernel ker 
        Both image and kernel's dimensions should be even: ker.shape % 2 == 0
    '''
    sy,sx = array(ker.shape)/2
    y0,x0 = array(im.shape)/2
    big_ker = zeros(im.shape)
    sy,sx,y0,x0 = int(sy), int(sx), int(y0), int(x0) 
    big_ker[y0-sy:y0+sy,x0-sx:x0+sx] = ker
    return irfft2(rfft2(im)*rfft2(fftshift(big_ker)))


def find_core(evt, sigma, R_search):
    im = ximage(evt, padding=True)
    # Both im and kernel sizes should be even.
    a = conv(im, gauss(sigma))
    y, x = indices(a.shape) - evt.d
    #figure(), imshow(a), show()
    #raise Exception
    return XY(maxxy(a*(y**2 + x**2 < R_search**2)) - evt.d)


def ellipsemask(xc,yc, a, b, th, inner=0):
    #assert a>0, b>0

    th = fmod(th, 180)/180*pi
    c = cos(th)
    s = sin(th)

    def func(y,x):
        x = x-float(xc)
        y = y-float(yc)
        x1 = c*x + s*y
        y1 = -s*x + c*y
        dist = x1**2/a**2 + y1**2/b**2
        return (inner**2 <= dist) & (dist < 1)

    return func


def kpc2pix(H, Om, z0):
    ''' Converts physical distances in kiloparsec to Chandra pixels at redshift z0
        H = Hubble constant in km/sec/Mpc
        Om = total matter density
    '''
    N = 64 # Number of evaluation points
    c = 3e5 # speed of light in km/s
    z = linspace(0, z0, N+1)
    E = sqrt( Om*(1+z)**3 + 1-Om )
    dC = 1e3 * c/H * romb(1/E, 1.*z0/N) # Comoving distance in Mpc
    dA = dC / (1+z0) # Angular diameter distance
    rad2arcsec = 180*3600/pi
    px2arcsec = 0.492
    return 1/dA * rad2arcsec / px2arcsec #kpc2px

# Cosmology ################################
#def M500toR500(M500, z):
#    Om = 0.3   # CCCP cosmology
#    rho_cr = 0.92e-26
#    Ez = sqrt( Om*(1+z)**3 + 1-Om )
#    rho = rho_cr * Ez**2
#    return (M500*1e14*2e30 / (500*rho*4*pi/3))**(1./3) / 3.1e16 / 1e3 # in kpc
#
#
#def Da(z):
#    #Schneider (4.47) p.158, CCCP cosmology
#    Om = 0.27
#    c = 3e5 # km/s
#    h = 0.7 # *100 km/(s*Mpc)
#    Da = c/(100*h) *2/Om**2/(1+z)**2 * (Om*z + (Om-2)*(sqrt(1+Om*z)-1)) # in Mpc
#    return Da
#
#def DL(z):
#    return (1+z)**2 * Da(z)
#
#def kpc2pix(H, Om, z):
#    rad2sec = 180*3600/pi
#    px2sec = 0.492
#    return 1e-3/Da(z) * rad2sec / px2sec # 1e-3 Mpc/kpc
#
#
