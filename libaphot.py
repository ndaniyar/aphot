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


from common import *


#### Exposure maps ######################################
def interp_search_vec(x0, a, l_init=0, r_init=0):
    ''' Interpolation search for each element of vector x0 in array a
        This function only works correctly if a[0] < x[i] <= a[-1] for any i
    '''
    bad_left = (a[0] > x0)
    if any(bad_left):
        print 'interp_search_vec error: not all(a[0] <= x0)'
        print 'a =', a
        print 'x0 =', x0
        print 'bad indices:', where(bad_left)[0]
        x0[bad_left] = a[0]
    #assert left

    bad_right = (x0 > a[-1])
    if any(bad_right):
        print 'interp_search_vec error: not all(x0 <= a[-1])'
        print 'a =', a
        print 'x0 =', x0
        print 'bad indices: ', bad_right
        x0[bad_right] == a[-1]
    #assert right

    l0 = empty_like(x0, int)
    r0 = empty_like(x0, int)
    l0.fill(l_init)
    r0.fill(r_init if r_init>0 else len(a)-1)

    compl = 0
    active = arange(len(x0))
    while len(active):
        l = l0[active]
        r = r0[active]
        x = x0[active]

        m = l + ( (x - a[l]) / (a[r] - a[l]) * (r-l) ).astype(int)

        m = maximum(m, l+1)
        m = minimum(m, r-1)

        x_less_than_mid = x < a[m]

        l = where(x_less_than_mid, l, m)  # a[m] < x  => left=m
        r = where(x_less_than_mid, m, r)  # x < a[m]  => right=m

        l0[active] = l        
        r0[active] = r

        active = active[r-l>1]

    return l0


def get_expm_funcs(expm, offset, bins, rads):
    ''' Given an exposure map (expm), cluster center (offset, in expm coordinates),
        bins (binning in expm), and annuli (rads), creates objects (expm_funcs) which approximate total
        exposure of all pixels integrated in a sector of polar angles from 0 to angle
        in each annulus
    '''
    ye, xe = indices(expm.shape, dtype=float) 
    xc = xe.ravel() * bins - offset.x
    yc = ye.ravel() * bins - offset.y
    r2_all = xc**2 + yc**2

    expm_funcs = []
    for i in range(len(rads)-1):
        minrad, maxrad = rads[i], rads[i+1]
        inside = (r2_all > (minrad - bins/2.0)**2) & (r2_all < (maxrad + bins/2.0)**2) 

        r = sqrt(r2_all[inside])
        phi = arctan2(yc[inside], xc[inside])/2.0/pi + 0.5
        dphi = 8/2/pi/r
        
        a = phi - dphi
        b = phi + dphi
        
        a_less_0 = a < 0
        b_greater_1 = b > 1
        crosses_zero = a_less_0 | b_greater_1
        sign = 1 - 2 * crosses_zero

        a[a_less_0] += 1
        b[b_greater_1] -= 1
        angles = hstack([0, a, b, 1])

        h = expm.flat[inside] * (minimum(r + bins/2.0, maxrad) - maximum(r - bins/2.0, minrad))
        zero_dens = sum(h[crosses_zero])
        ddens = hstack([zero_dens, sign*h, -sign*h, -zero_dens])

        order = angles.argsort()
        angles = angles[order]
        dens = cumsum(ddens[order])

        dangles = ediff1d(angles, to_end=0)
        distr = roll(cumsum(dangles*dens), 1)
        norm = distr[0]
        distr[0] = 0

        expm_funcs += [sdict(angles=angles, dens=dens, distr=distr, norm=norm)]

    return expm_funcs


def rad_exposure(r, expm, offset, bins):
    ''' Given an exposure map (expm), cluster center (offset, in expm coordinates),
        and a set of radii (r), computes total exposure of all pixels within each 
        radius r
    '''
    ye, xe = indices(expm.shape) 
    xc = xe.ravel() * bins - offset.x
    yc = ye.ravel() * bins - offset.y
    re = sqrt(xc**2 + yc**2)
    sortind = re.argsort()
    re = re[sortind]
    rexp = cumsum(expm[ ye.ravel()[sortind], xe.ravel()[sortind] ])
    re[0] = 0 # To satisfy interp_search_vec requirement 
    ind = interp_search_vec(r, re)
    return bins**2 * rexp[ind]


#### Helper functions for adaptive annuli aphot
def pav(y, nonnegative=False):
    """
    PAV uses the pair adjacent violators method to produce a monotonic
    smoothing of y
    translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
    DN: added nonnegative option (2013)
    """
    y = np.asarray(y)
    assert y.ndim == 1
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    while True:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break
 
        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]
 
        val = s / n
        if nonnegative and val<0: val=0
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
    return v
 

def movave(a, w=10):
    if 2*w+1 > len(a): w = int((len(a)-1)/2) # if len(ker) > len(a), convolve returns greater length array
    ker = ones(2*w+1)
    ker[w] = 0
    b = convolve(a, ker, mode='same') 
    w = convolve(ones_like(a), ker, mode='same')
    return b/w


#### Asymmetry ############################################
def watson_test(x, emdf=None):
    n = len(x)
    if n==0: raise Exception #return 0
    x = sort(x)
    if emdf is not None:
        ang = interp_search_vec(x, emdf.angles)
        f = emdf.distr[ang] + emdf.dens[ang]*(x-emdf.angles[ang])
        f /= emdf.norm
    else:
        f = x
    return sum((arange(0.5,n)/n - f)**2) + 1./12/n - (sum(f) - n/2.)**2/n


def watson_distance(x, emdf=None):
    if len(x)==0: return 0
    return (watson_test(x, emdf) - 1./12)/len(x)


def aphot(evt, center, rads, expm=None, adaptive_annuli=0): 
    ''' Computes A_phot 
    '''
    rads = array(rads)

    xcc = evt.xc-center.x
    ycc = evt.yc-center.y
    r = sqrt(xcc**2 + ycc**2)
    phi = arctan2(ycc, xcc)/2/pi + 0.5
    
    sortind = r.argsort()
    r = r[sortind]
    phi = phi[sortind]

    maxrad_ind = interp_search_vec(arr(1.1*rads[-1]), r)
    r = r[:maxrad_ind]
    phi = phi[:maxrad_ind]

    if adaptive_annuli:
        eff_area = rad_exposure(r, expm.map, expm.ofs+center) if expm is not None else pi*r**2
        cfl = arange(len(r)) - eff_area*evt.bkgr
        i1, i2 = interp_search_vec(arr(rads[0],rads[-1]), r)
        smooth_cfl = pav(movave(cfl, 30), nonnegative=True)
        smooth_cfl[0] = 0 # interp_search_vec requirement
        fluxes = linspace(smooth_cfl[i1], smooth_cfl[i2], adaptive_annuli+1).astype(int)
        ann = interp_search_vec(fluxes, smooth_cfl)
        rads = r[ann]
    else:
        ann = interp_search_vec(rads, r)

    Nrings = len(rads)-1
    emdf = get_expm_funcs(expm.map, expm.ofs+center, expm.bins, rads) if expm is not None else [None]*Nrings
    wats, asym, tot_counts, cl_counts, weight, radweight = (zeros(Nrings) for _ in range(6))
    
    eff_area = rad_exposure(rads, expm.map, expm.ofs+center, expm.bins) if expm is not None else pi*rads**2
    B = diff(eff_area)*evt.bkgr
    
    for i in range(Nrings):
        lo, hi = ann[i:i+2]
        N = hi - lo
        C = N - B[i]

        wats[i] = watson_distance(phi[lo:hi], emdf[i])
        asym[i] = (N/C)**2 * wats[i]
        weight[i] = C
        tot_counts[i] = N
        cl_counts[i] = C

    aphot = 100 * sum(asym*cl_counts)/sum(cl_counts)

    return sdict( annuli = rads, total_counts = tot_counts, cluster_counts = cl_counts,
                  wats = wats, center = center, aphot = aphot)

