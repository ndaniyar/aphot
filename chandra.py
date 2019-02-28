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


from scipy.misc import imsave
import numpy as np 
import astropy.io.fits
from common import *


def radec2xy(ra, dec, h):
    ''' Given a header of Chandra level 2 files, converts celestial coordintates
        to Chandra SKY coordinate system
    '''
    try:
        res = XY((ra-h['TCRVL11'])/h['TCDLT11']*cos(h['TCRVL12']/180*pi) + h['TCRPX11'] - 0.5,
              (dec-h['TCRVL12'])/h['TCDLT12'] + h['TCRPX12'] - 0.5)
    except KeyError:
        res = XY((ra-h['CRVAL1'])/h['CDELT1']*cos(h['CRVAL2']/180*pi) + h['CRPIX1'] - 0.5,
              (dec-h['CRVAL2'])/h['CDELT2'] + h['CRPIX2'] - 0.5)
    return res
        

def radec2xy_winding(ra, dec, h):
    ''' Same as radec2xy, but resolves uncertainty around ra=0
    '''
    x, y = radec2xy(ra, dec, h) 
    if x>10000: 
        x, y = radec2xy(ra+360, dec, h) 
    elif x<-10000: 
        x, y = radec2xy(ra-360, dec, h)
    return XY(x, y)


def xy2radec(x, y, h):
    ''' Given a header of Chandra level 2 files, converts celestial coordintates
        to Chandra SKY coordinate system
    '''
    return XY( (x-h['TCRPX11']+0.5) * h['TCDLT11'] / cos(h['TCRVL12']/180*pi) + h['TCRVL11'],
               (y-h['TCRPX12']+0.5) * h['TCDLT12']  + h['TCRVL12'] )

def read_regions(filename, header):
    ''' Read regions file in ds9 format
    '''
    regions = []
    for line in open(filename):
        if line.startswith('circle('):
            par = line.strip(')"\n').lstrip('circle(').split(',')
            ra, dec, rad = [float(x) for x in par]
            a = b = int(ceil(rad / 0.492))
            theta = 0
        elif line.startswith('ellipse('):
            par = line.strip(')"\n').lstrip('ellipse(').split(',')
            ra, dec, a, b, theta = [float(x.strip('"')) for x in par]
            a = int(ceil(a / 0.492))
            b = int(ceil(b / 0.492))
        else:
            continue

        x, y = radec2xy_winding(ra, dec, header) 

        x = int(round(x))
        y = int(round(y))
        regions += [Ellipse(x,y,a,b,theta)]
    return regions


def mask_regions(mask, regions, offset):
    ''' Mask point sources
    '''
    y_ind, x_ind = indices(mask.shape)
    for reg in regions:
        xc, yc, a, b, th = reg
        xc -= offset.x
        yc -= offset.y
        r = max(a,b)
        xl, xr = xc - r, xc + r + 1
        yl, yr = yc - r, yc + r + 1
        if xl < 0: xl = 0
        if xr < 0: xr = 0
        if yl < 0: yl = 0
        if yr < 0: yr = 0
        yl,yr,xl,xr = int(yl), int(yr), int(xl), int(xr) 
        mask[yl:yr, xl:xr] &= ~ellipsemask(xc, yc, a, b, th)(y_ind[yl:yr,xl:xr], x_ind[yl:yr,xl:xr])


def generate_uniform_exposure_map(evt_file):
    ''' Generates a fake exposure map assuming uniform illumitation of the chip
    '''
    def roll(x, y, alpha):
        s = sin(alpha/180*pi)
        c = cos(alpha/180*pi)
        return x*c - y*s, x*s + y*c
    
    hdu = astropy.io.fits.open(evt_file) 
    data = hdu[1].data
    header = hdu[1].header
    
    ind = 1.*arange(1025)*8 + 0.5
    exph = {'CRPIX1': 512.5, 'CRPIX2': 512.5, 'CDELT1':-1.093e-3} 
    expm = zeros((1025,1025))

    xi, yi = roll(*meshgrid(ind, ind), alpha = header['ROLL_NOM'])
    xe, ye = roll(data.x, data.y, alpha = header['ROLL_NOM'])

    expm[(xe.min() < xi) & (xi < xe.max()) & (ye.min() < yi) & (yi < ye.max())] = 1

    return exph, expm


def find_point_sources(evt):
    ''' Finds point sources if region file is not provided.
        This function is known to miss some point sources and other
        artefacts in Chandra files. Us it at your own risk
    '''
    offset = -XY(min(evt.x), min(evt.y)) + b
    imsize = (ptp(evt.x)+2*b, ptp(evt.y)+2*b)                

    regions = []
    # make image
    im0 = zeros(imsize, float)
    for x,y in zip(evt.x,evt.y): im0[y+offset.y,x+offset.x] +=1
    # Convolutions
    a = array([conv(im0, ker) for ker in kernels])
    for i in ind(evt.x):
        v = a[:, evt.y[i]+offset.y, evt.x[i]+offset.x]
        pwr = v[3:]/v[:-3]
        z, m = argmin(pwr), min(pwr)
        if (3 < z < len(pwr)-1 and m < 2 and v[z]>3): # these are point sources
            r = kersize[z]
            regions += [Ellipse(evt.x[i], evt.y[i], r, r, 0)]
    
    return regions


def get_events_expmap(evt, exph, expm, regions, ths, R_bkgr_px, inspect=False):
    ''' Alignes event files, exposure maps and region files. 
        Excludes point sources from event files and exposure maps.
        Estimates backgound. 
        Saves aligned images in a .png with a prefix given in inspect
    '''
    # Load chandra events
    center = evt.center
    CCD = expm > expm.max()/5
    expm[~CCD] = 0
    expm_norm = mean(expm[CCD])


    bins = int(round(exph['CDELT1']/evt.header['TCDLT11']))
    binpx = ones((bins,bins), dtype=bool) 
    # Subtract offset to convert evt coordinates to exp coordinates 
    offset = -radec2xy_winding(*xy2radec(0, 0, evt.header), h=exph) * bins
    evt.y_expm = intround(evt.y - offset.y)
    evt.x_expm = intround(evt.x - offset.x)

    # Mask point sources
    ptsrc_mask = kron(ones_like(expm, dtype=bool), binpx)
    mask_regions(ptsrc_mask, regions, offset)

    # Find borders
    expm_mask = expm.astype(bool)
    x_ind = nonzero(any(expm_mask, 0))[0]
    xl, xr = x_ind[0], x_ind[-1]+1
    y_ind = nonzero(any(expm_mask, 1))[0]
    yl, yr = y_ind[0], y_ind[-1]+1
    xlb, xrb, ylb, yrb = arr(xl, xr, yl, yr)*bins

    # Mask annulus for background estimation
    bkgr_mask = kron(expm > 0, binpx)
    msize = bkgr_mask.shape[0]
    x = arange(xlb, xrb)
    y = arange(ylb, yrb)
    rad2 = outer(ones_like(y),(x-center.x)**2) + outer((y-center.y)**2,ones_like(x))
    bkgr_mask[ylb:yrb, xlb:xrb] &= (R_bkgr_px[0]**2 < rad2) & (rad2 < R_bkgr_px[1]**2) 
    del rad2

    # Save images for visual control
    if inspect:
        show_counts = kron(zeros_like(expm, dtype=int8), binpx)
        show_counts[evt.y_expm, evt.x_expm] += 1

        expm_patch = expm[yl:yr,xl:xr]
        expm_patch /= expm_patch.max() 

        imsave(inspect+'.png', 
                dstack((kron(expm_patch, binpx) * ptsrc_mask[ylb:yrb, xlb:xrb],
                        show_counts[ylb:yrb, xlb:xrb],
                        bkgr_mask[ylb:yrb, xlb:xrb]))[::-1,:,:])

    # Exclude point sources from evt
    u = ptsrc_mask[evt.y_expm, evt.x_expm]  #& (bkgr_mask[evt.y,evt.x] == 0)
    evt.x = evt.x[u]
    evt.y = evt.y[u]
    evt.x_expm = evt.x_expm[u]
    evt.y_expm = evt.y_expm[u]
    tmp = expm[intround((evt.y-offset.y)/bins), intround((evt.x-offset.x)/bins)]

    # Exclude off-chip counts
    u = (tmp > 0)
    evt.x = evt.x[u]
    evt.y = evt.y[u]
    evt.w = 1/tmp[u]

    # Asymmetry background
    bkgr_counts = sum(bkgr_mask[evt.y_expm, evt.x_expm])
    bkgr_exp = sum(kron(expm, binpx)[ptsrc_mask & bkgr_mask])
    
    # Power ratios background
    #bkgr_flux = sum(1./evt.w)
    #bkgr_area = sum(ptsrc_mask & bkgr_mask)
    #bkgr_pr = bkgr_flux / bkgr_area 

    # Finalize evt data structure
    evt.xc = intround(evt.x - center.x)
    evt.yc = intround(evt.y - center.y)
    del evt.x
    del evt.y
    del evt.x_expm
    del evt.y_expm
    u = (abs(evt.xc) < ths) & (abs(evt.yc) < ths)
    evt.xc = evt.xc[u]
    evt.yc = evt.yc[u]
    evt.w = evt.w[u]
    evt += sdict(bkgr_counts = bkgr_counts, bkgr_exp = bkgr_exp) 

    # Cut expm
    xl = round((-ths+center.x-offset.x)/bins)
    yl = round((-ths+center.y-offset.y)/bins)
    xr = xl + round(2.*ths/bins) + 1
    yr = yl + round(2.*ths/bins) + 1
    yl,yr,xl,xr = int(yl), int(yr), int(xl), int(xr) 


    expm = expm[yl:yr, xl:xr]
    tmp = ptsrc_mask[yl*bins:yr*bins, xl*bins:xr*bins].astype(int)
    ptsrc_exp = sum(sum(tmp.reshape(yr-yl, bins, xr-xl, bins), 3), 1)
    expm *= ptsrc_exp
    expm = expm/(bins**2) 


    ofs = center - offset - bins*arr(xl, yl)
    cl_expm = sdict(map = expm, ofs = ofs, bins=bins, norm = expm_norm)

    return evt, cl_expm


def load_evt_file(evt_file, ra, dec, Eband, evt=None):
    # Load chandra events
    hdu = astropy.io.fits.open(evt_file)
    data = hdu[1].data
    header = hdu[1].header

    if evt is None:
        offset = XY(0,0)
        evt = sdict(x = arr(), y = arr())
        evt.header = header 
        evt.center = radec2xy_winding(ra, dec, evt.header) # cluster center in evt coordinates
    else:
        offset = -radec2xy_winding(*xy2radec(0, 0, header), h=evt.header)

    u = (Eband[0] < data.energy) & (data.energy < Eband[1])
    evt.x = hstack([evt.x, data.x[u] - offset.x])
    evt.y = hstack([evt.y, data.y[u] - offset.y])
    return evt


def process_multiple_obs(ra, dec, ths, Eband, R_bkgr_px, 
                            path, evt_files, exp_files, reg_files, inspect=None):
    no_region_files = 'Region files are not provided. '\
                      'Trying to find point sources. '\
                      'This will take some time.'

    if len(exp_files) == 1:
        # Get exposure map
        hdu = astropy.io.fits.open(exp_files[0]) 
        exph = hdu[0].header
        expm = hdu[0].data


        # Go through the event files in reverse, to remember the header of the
        # first file when done. The coordinate system defined in this header is
        # used throughout data loading process
        evt = None
        for evt_file in evt_files: 
            evt = load_evt_file(path+evt_file, ra, dec, Eband, evt)

        # get all regions
        if len(reg_files) == 0: # no reg files, try to identify point sources
            print (no_region_files)
            regions = find_point_sources(evt)
        else:
            regions = []
            for reg_file in reg_files:
                regions.extend(read_regions(path+reg_file, evt.header))

        evt, expm = get_events_expmap(evt, exph, expm, regions, ths, R_bkgr_px, inspect)
        del evt.header

        expm.ofs = intround(expm.ofs)
        evt.bkgr_exp = evt.bkgr_exp/expm.norm

    else:
        if len(exp_files) == 0: 
            exp_files = [None]*len(evt_files)
        if len(exp_files) != len(evt_files): 
            raise Exception('Mismatch between event files and exposure maps')

        if len(reg_files) == 0: 
            reg_files = [None]*len(evt_files)
        if len(reg_files) != len(reg_files): 
            raise Exception('Mismatch between event files and point sources files')

        evt_l = []
        expm_l = []
        for evt_file, exp_file, reg_file in zip(evt_files, exp_files, reg_files):
            evt = load_evt_file(path+evt_file, ra, dec, Eband)

            # Get next exposure map
            if exp_file is None:
                exph, expm = generate_uniform_exposure_map(evt_file)
            else:
                hdu = astropy.io.fits.open(path+exp_file) 
                exph = hdu[0].header
                expm = hdu[0].data

            # Get next region file
            if reg_file is None:
                print (no_region_files)
                regions = find_point_sources(evt)
            else:
                regions = read_regions(path+reg_file, evt.header)

            # Align the next set of event file, exposure map, and region file
            inspect1 = None
            if inspect: inspect1 = path + inspect + str(len(evt_l)+1) + '.png'
            evt, expm = get_events_expmap(evt, exph, expm, regions, ths, R_bkgr_px, inspect1)
            evt_l += [evt]
            expm_l += [expm]

        # Combine evt and expm for all observations
        expm = sdict()
        expm.map = sum([x.map for x in expm_l],0)
        expm.ofs = XY(intround(mean([x.ofs for x in expm_l],0)))
        expm.bins = expm_l[0].bins
        expm.norm = sum([x.norm for x in expm_l])

        evt = sdict()
        evt.xc = hstack([v.xc for v in evt_l])
        evt.yc = hstack([v.yc for v in evt_l])
        evt.w = hstack([v.w for v in evt_l])
        evt.bkgr_counts = sum([v.bkgr_counts for v in evt_l])
        evt.bkgr_exp = sum([v.bkgr_exp for v in evt_l]) / expm.norm

    if inspect is not None:
        r = kron(expm.map, ones((expm.bins,expm.bins)))
        g = zeros(array(expm.map.shape)*expm.bins)
        for xc, yc in zip(evt.xc, evt.yc):
            g[yc + expm.ofs.y, xc + expm.ofs.x] += 1
        g[g>3] = 3    
        r *= g.max()/r.max()/3
        imsave(path + inspect + '_all_obs.png', 
               dstack([r, g, zeros(array(expm.map.shape)*expm.bins)])[::-1,:,:]) 

    b = 102 # classify_events legacy
    evt += sdict(ths=ths, b=b, d=ths+b, ims=2*(ths+b))
    evt.bkgr = evt.bkgr_counts / evt.bkgr_exp
    expm.map = expm.map/expm.norm 
    
    return evt, expm

