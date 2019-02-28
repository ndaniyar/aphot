# This file is a part of library for computing Photon Asymmetry (A_phot)
# parameter for morphological classification of cluster.
# This parameter is described in detail in the following publication:
# 
#     xxxxxxxx
#
# This program is free software: you can redistribute it, modify
# it or unclude parts of it in your free or proprietary software, provided
# that you reference the aforementioned scientific publication.
#
# Copyright (C) 2013 Daniyar Nurgaliev

from numpy.random import RandomState
from scipy.stats import scoreatpercentile
from sys import stdout

from common import sdict, find_core, kpc2pix
from get_args import get_args
from chandra import process_multiple_obs
from libaphot import aphot


# Default parameters
p = sdict( 
    H = 70,
    Om = 0.27,
    Eband = [500, 5000],        # Energy filter: Eband[0] < E < Eband[1]
    R_bkgr = [2, 4],            # Annulus for background estimation in units of R500 
    annuli = [0.05, 0.12, 0.2, 0.3, 1.0], # Annuli for A_phot
    inspect = None,             # Filename prefix for visual control images
    Nresamplings = 100,
    path = '',
    evt_files = '',
    exp_files = '',
    reg_files = ''
          )

# Get parameters from the command line and configuration file
p.update(get_args())

kpc2px = kpc2pix(p.H, p.Om, p.z) # kpc to pixels conversion factor
R500px = p.R500 * kpc2px # R500 in pixels

# Convert multiple observations, exposure maps, and region files into
# internal representation of X-ray events (evt) and exposure map (expm)
evt, expm = process_multiple_obs(p.ra, p.dec, int(1.5*R500px), 
                    p.Eband, [x*R500px for x in p.R_bkgr],
                    p.path, p.evt_files.split(), p.exp_files.split(), p.reg_files.split(),
                    p.inspect)

print ('Calculating Aphot ...')
stdout.flush()

# Find cluster core as the brightest point after convolution with
# sigma = 40 Mpc Gaussian, within 400 Mpc from cluster center coordinates
center = find_core(evt, 40*kpc2px, 400*kpc2px)

# Calculate A_phot
annuli_px = [x*R500px for x in p.annuli]
ap0 = aphot(evt, center, annuli_px, expm=expm)


# Calculate uncertainty ###########################################3

def sample_half_counts(evt0, seed=0):
    evt = sdict(b = evt0.b, d = evt0.d, ims = evt0.ims, ths = evt0.ths,
                bkgr = evt0.bkgr/2)
    N = len(evt0.xc)
    rnd = RandomState(seed)
    m = rnd.randint(low=0, high=N, size=rnd.binomial(N,0.5))

    evt.xc = evt0.xc[m]
    evt.yc = evt0.yc[m]
    evt.w = evt0.w[m]

    return evt


if p.Nresamplings > 0:
    print ('\nCalculating uncertainty ...')
    ap = []
    for r in range(p.Nresamplings): 
        stdout.flush()
        evt1 = sample_half_counts(evt, seed=r)
        center = find_core(evt1, 40*kpc2px, 400*kpc2px)
        ap.append( aphot(evt, center, annuli_px, expm=expm).aphot )

    lo  = scoreatpercentile(ap, 16)
    mid = scoreatpercentile(ap, 50)
    hi  = scoreatpercentile(ap, 84)

    print ('\nAphot =', ap0.aphot, '+', (mid-lo)/1.4142, '-', (hi-mid)/1.4142 )
else:
    print ('\nAphot =', ap0.aphot)
