import numpy as np, sys, os, scipy as sc
import scipy.ndimage as ndimage
import scipy.integrate as integrate
from pylab import *
try:
    import healpy as H
except:
    cmd = 'pip install healpy'
    os.system(cmd)
    import healpy as H

################################################################################################################
#flat-sky routines
################################################################################################################

def cl_to_cl2d(el, cl, flatskymapparams):

    """
    converts 1d_cl to 2d_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl

    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape) 

    return cl2d

################################################################################################################

def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    output:
    lx, ly
    """

    nx, ny, dx, dx = flatskymapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

################################################################################################################

def get_lxly_az_angle(lx,ly):

    """
    azimuthal angle from lx, ly

    inputs:
    lx, ly = 2d lx and ly arrays

    output:
    azimuthal angle
    """
    return 2*np.arctan2(lx, -ly)

################################################################################################################
def convert_eb_qu(map1, map2, flatskymapparams, eb_to_qu = 1):

    lx, ly = get_lxly(flatskymapparams)
    angle = get_lxly_az_angle(lx,ly)

    map1_fft, map2_fft = np.fft.fft2(map1),np.fft.fft2(map2)
    if eb_to_qu:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft - np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real
    else:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft + np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( -np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real

    return map1_mod, map2_mod

################################################################################################################
def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type = 0):
    """
    filter_type = 0 - low pass filter
    filter_type = 1 - high pass filter
    filter_type = 2 - band pass
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        fft_filter[ell>lmin_lmax] = 0.
    elif filter_type == 1:
        fft_filter[ell<lmin_lmax] = 0.
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell<lmin] = 0.
        fft_filter[ell>lmax] = 0

    return fft_filter
################################################################################################################

def wiener_filter(mapparams, cl_signal, cl_noise, el = None):

    if el is None:
        el = np.arange(len(cl_signal))

    nx, ny, dx, dx = flatskymapparams

    #get 2D cl
    cl_signal2d = cl_to_cl2d(el, cl_signal, flatskymapparams) 
    cl_noise2d = cl_to_cl2d(el, cl_noise, flatskymapparams) 

    wiener_filter = cl_signal2d / (cl_signal2d + cl_noise2d)

    return wiener_filter

################################################################################################################
def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None, minbin = 0, maxbin = 10000, mask = None, filter_2d = None, return_2d = False):

    """
    map2cl module - get the power spectra of map/maps

    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.

    flatskymap1: map1 with dimensions (ny, nx)
    flatskymap2: provide map2 with dimensions (ny, nx) cross-spectra

    binsize: el bins. computed automatically if None

    cross_power: if set, then compute the cross power between flatskymap1 and flatskymap2

    output:
    auto/cross power spectra: [el, cl, cl_err]
    """

    nx, ny, dx, dx = flatskymapparams
    dx_rad = np.radians(dx/60.)

    lx, ly = get_lxly(flatskymapparams)

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)

    if filter_2d is not None:
        flatskymap_psd = flatskymap_psd / filter_2d

    if return_2d:
        el = None
        cl = flatskymap_psd
    else:
        #rad_prf = radial_profile_v1(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
        #el, cl = rad_prf[:,0], rad_prf[:,1]
        el, cl = radial_profile(flatskymap_psd, binsize, maxbin, minbin=minbin, xy=(lx,ly), return_errors=False)

    if mask is not None:
        fsky = np.mean(mask)
        cl /= fsky

    '''
    if filter_2d is not None:
        if not return_2d:
            #rad_prf_filter_2d = radial_profile_v1(filter_2d, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
            #el, fl = rad_prf_filter_2d[:,0], rad_prf_filter_2d[:,1]
            el, fl = radial_profile(filter_2d, binsize, maxbin, minbin=minbin, xy=(lx,ly), return_errors=False)
        else:
            fl = filter_2d
        cl /= fl
    '''

    return el, cl

################################################################################################################

def make_gaussian_realisation(mapparams, el, cl, cl2 = None, cl12 = None, bl = None, qu_or_eb = 'eb'):

    nx, ny, dx, dy = mapparams
    arcmins2radians = np.radians(1/60.)

    dx *= arcmins2radians
    dy *= arcmins2radians

    ################################################
    #map stuff
    norm = np.sqrt(1./ (dx * dy))
    ################################################

    #1d to 2d now
    cltwod = cl_to_cl2d(el, cl, mapparams)
    
    ################################################
    if cl2 is not None: #for TE, etc. where two fields are correlated.
        assert cl12 is not None
        cltwod12 = cl_to_cl2d(el, cl12, mapparams)
        cltwod2 = cl_to_cl2d(el, cl2, mapparams)

    ################################################
    if cl2 is None:

        cltwod = cltwod**0.5 * norm
        cltwod[np.isnan(cltwod)] = 0.

        gauss_reals = np.random.standard_normal([ny,nx])
        SIM = np.fft.ifft2( np.copy( cltwod ) * np.fft.fft2( gauss_reals ) ).real

    else: #for TE, etc. where two fields are correlated.

        cltwod12[np.isnan(cltwod12)] = 0.
        cltwod2[np.isnan(cltwod2)] = 0.

        gauss_reals_1 = np.random.standard_normal([ny,nx])
        gauss_reals_2 = np.random.standard_normal([ny,nx])


        gauss_reals_1_fft = np.fft.fft2( gauss_reals_1 )
        gauss_reals_2_fft = np.fft.fft2( gauss_reals_2 )

        #field_1
        cltwod_tmp = np.copy( cltwod )**0.5 * norm
        SIM_FIELD_1 = np.fft.ifft2( cltwod_tmp *  gauss_reals_1_fft ).real
        #SIM_FIELD_1 = np.zeros( (ny, nx) )

        #field 2 - has correlation with field_1
        t1 = np.copy( gauss_reals_1_fft ) * cltwod12 / np.copy(cltwod)**0.5
        t2 = np.copy( gauss_reals_2_fft ) * ( cltwod2 - (cltwod12**2. /np.copy(cltwod)) )**0.5
        SIM_FIELD_2_FFT = (t1 + t2) * norm
        SIM_FIELD_2_FFT[np.isnan(SIM_FIELD_2_FFT)] = 0.
        SIM_FIELD_2 = np.fft.ifft2( SIM_FIELD_2_FFT ).real

        #T and E generated. B will simply be zeroes.
        SIM_FIELD_3 = np.zeros( SIM_FIELD_2.shape )
        if qu_or_eb == 'qu': #T, Q, U: convert E/B to Q/U.
            SIM_FIELD_2, SIM_FIELD_3 = convert_eb_qu(SIM_FIELD_2, SIM_FIELD_3, mapparams, eb_to_qu = 1)
        else: #T, E, B: B will simply be zeroes
            pass

        SIM = np.asarray( [SIM_FIELD_1, SIM_FIELD_2, SIM_FIELD_3] )


    if bl is not None:
        if np.ndim(bl) != 2:
            bl = cl_to_cl2d(el, bl, mapparams)
        SIM = np.fft.ifft2( np.fft.fft2(SIM) * bl).real

    SIM = SIM - np.mean(SIM)

    return SIM

################################################################################################################

def radial_profile(image, binsize, maxbin, minbin=0.0, xy=None, return_errors=False):
    """
    Get the radial profile of an image (both real and fourier space)

    Parameters
    ----------
    image : array
        Image/array that must be radially averaged.
    binsize : float
        Size of radial bins.  In real space, this is
        radians/arcminutes/degrees/pixels.  In Fourier space, this is
        \Delta\ell.
    maxbin : float
        Maximum bin value for radial bins.
    minbin : float
        Minimum bin value for radial bins.
    xy : 2D array
        x and y grid points.  Default is None in which case the code will simply
        use pixels indices as grid points.
    return_errors : bool
        If True, return standard error.

    Returns
    -------
    bins : array
        Radial bin positions.
    vals : array
        Radially binned values.
    errors : array
        Standard error on the radially binned values if ``return_errors`` is
        True.
    """

    image = np.asarray(image)
    if xy is None:
        y, x = np.indices(image.shape)
    else:
        y, x = xy

    radius = np.hypot(y, x)
    radial_bins = np.arange(minbin, maxbin, binsize)

    hits = np.zeros(len(radial_bins), dtype=float)
    vals = np.zeros_like(hits)
    errors = np.zeros_like(hits)

    for ib, b in enumerate(radial_bins):
        inds = np.where((radius >= b) & (radius < b + binsize))
        imrad = image[inds]
        total = np.sum(imrad != 0.0)
        hits[ib] = total

        if total > 0:
            # mean value in each radial bin
            vals[ib] = np.sum(imrad) / total
            errors[ib] = np.std(imrad)

    bins = radial_bins + binsize / 2.0

    std_mean = np.sum(errors * hits) / np.sum(hits)
    
    if return_errors:
        errors = std_mean / hits ** 0.5
        return bins, vals, errors
    else:
        return bins, vals

################################################################################################################

def get_filter(els, cl_signal = None, cl_noise = None, ksz_dl_power = 3., lmin_for_filter = 3000., lmax_for_filter = 5000., mapparams = None, use_sqroot_signal = False):
    ##if cl_signal is None and cl_noise is None:
    if cl_signal is None:
        cl_signal = get_ksz_power_for_filtering(els, ksz_dl_power = ksz_dl_power)

    if cl_noise is None:
        wiener_filter = np.ones_like( els ) * 1.
    else:
        if not use_sqroot_signal:
            wiener_filter = cl_signal / np.asarray( cl_noise )
        else:
            wiener_filter = np.sqrt( cl_signal ) / np.asarray( cl_noise )
    wiener_filter[els<lmin_for_filter] = 0.
    wiener_filter[els>lmax_for_filter] = 0.
    wiener_filter[np.isinf(wiener_filter)] = 0.; 
    wiener_filter[np.isnan(wiener_filter)] = 0.
    wiener_filter /= np.max(wiener_filter)

    if mapparams is not None:
        wiener_filter_2D = cl_to_cl2d(els, wiener_filter, mapparams)
        return wiener_filter, wiener_filter_2D
    else:

        ##plot(els, wiener_filter); show()
        return wiener_filter

################################################################################################################

def get_kbar(tmpels, tmpwls, tmpcls_signal = None, tmpmap = None, angres_am = 0.5, mask = None, binsize = 50., lmax = 10000., elmin = 2500., elmax = 7000., ell_bins = 500, filter_only = True):

    if not filter_only:
        assert tmpcls_signal is not None or tmpmap is not None
        if tmpcls_signal is None:
            assert tmpmap is not None
            ny, nx = tmpmap.shape
            mapparams = [ny, nx, angres_am, angres_am]
            el_, tmpcls_signal = map2cl(mapparams, tmpmap, binsize = binsize, maxbin = lmax, mask = mask)
            tmpcls_signal = np.interp(tmpels, el_, tmpcls_signal)


    def integrand(ln_ell):
        ell = np.exp(ln_ell)
        wl = np.interp(ell, tmpels, tmpwls)
        if (0):
            plot(tmpels, tmpwls, lw = 2.)
            plot(ell, wl, color = 'orangered')
            show(); sys.exit()
        if filter_only:
            return wl#**2
        else:
            cl = np.interp(ell, tmpels, tmpcls_signal)
            dl_fac = ell * (ell+1)/2/np.pi
            dl = dl_fac * cl
            return dl * wl**2

    ln_ells   = np.linspace(np.log(elmin), np.log(elmax), ell_bins)
    kbar = integrate.simps( integrand(ln_ells), x = ln_ells )
    #kbar = integrate.quad(integrand, lmin, lmax)[0]    
    return kbar

################################################################################################################

def get_ksz_power_for_filtering(els, ksz_dl_power = 3.):
    dl_ksz = ( np.ones_like(els) + ksz_dl_power ) #uK^2
    dl_fac = els * (els+1)/2/np.pi
    cl_ksz = np.zeros_like(dl_ksz)
    #20231009
    ##cl_ksz[dl_fac!=0] = dl_ksz[dl_fac!=0] / els[dl_fac!=0]
    cl_ksz[dl_fac!=0] = dl_ksz[dl_fac!=0] / dl_fac[dl_fac!=0]
    return cl_ksz

################################################################################################################

def get_mask(ny, nx, angres_am = 0.5, mask_deg = 5., circle_or_square = 'sqaure'):
    npix_cos=60. #int( 10. * 60. / map_resol_arcmins )+2
    ##print(smap.shape); sys.exit()
    boxsize_deg = ny * (angres_am)/60.
    ra_arr = dec_arr = np.linspace(-boxsize_deg/2, boxsize_deg/2, ny)
    ra_grid, dec_grid = np.meshgrid(ra_arr, dec_arr)
    radius_grid = np.hypot(ra_grid, dec_grid)
    newmask = np.ones( (ny, nx) )
    if circle_or_square == 'circle':
        newmask[radius_grid>mask_deg]= 0.
    elif circle_or_square == 'sqaure':
        inds = np.where( (abs(ra_grid)>mask_deg) | (abs(dec_grid)>mask_deg) )
        newmask[inds]= 0.
    ##imshow( newmask ); colorbar(); show(); sys.exit()
    hanning=np.hanning(npix_cos)
    hanning=np.sqrt(np.outer(hanning,hanning))
    newmask=ndimage.convolve(newmask, hanning)
    newmask/=np.max(newmask)

    return newmask

################################################################################################################

def remove_mean_and_mask(smap, mask = None, add_new_mask = False, angres_am = 0.5, circular_mask_deg = 5.):
    if mask is None:
        mask = np.ones_like( smap )

    mean_val = np.sum(smap) / np.sum(mask)
    smap = (smap - mean_val) * mask #remove mean
    if add_new_mask:
        newmask = get_circular_mask(smap, angres_am = angres_am, circular_mask_deg = circular_mask_deg)
        mask = np.copy(newmask)
        smap = smap * mask

    return smap, mask

################################################################################################################

def filter_alm(alm, wl):
    #filter alm
    alm_filtered = H.almxfl(alm, wl)
    
    return alm_filtered

################################################################################################################
    
def get_map_from_alm(els, fname, wl = None, nside = 512):
    
    #read alm
    curr_alm = H.read_alm(fname)
    
    #get Gaussianised alm
    curr_cl_unfiltered = H.alm2cl(curr_alm)
    
    if wl is not None:
        curr_alm = filter_alm(curr_alm, wl)        
        
    #get Gaussianised alm
    curr_cl = H.alm2cl(curr_alm)
    curr_alm_gaussianised = H.synalm( curr_cl )
    
    #get maps now
    curr_hmap = H.alm2map( curr_alm, nside = nside )
    curr_hmap_gaussianised = H.alm2map( curr_alm_gaussianised, nside = nside )    
    
    return curr_cl_unfiltered, curr_hmap, curr_hmap_gaussianised  
    
################################################################################################################

def get_4pt_flatsky(smap, smap2 = None, mask = None, binsize = 50, bigk_lmax = 1000, angres_am = 0.5, add_new_mask_for_big_K = False):

    if smap2 is None:
        smap2 = np.copy(smap)

    #remove mean and mask
    smap, mask_mod = remove_mean_and_mask(smap, mask = mask, add_new_mask = False)
    smap2, mask_mod = remove_mean_and_mask(smap2, mask = mask, add_new_mask = False)

    bigK = smap * smap2

    #remove mean and mask
    bigK, mask_mod = remove_mean_and_mask(bigK, mask = mask, add_new_mask = add_new_mask_for_big_K, angres_am = angres_am)
    
    #now get the power spectrum of the bigK map
    ny, nx = smap.shape
    mapparams = [ny, nx, angres_am, angres_am]
    el_, cl_big_k = map2cl(mapparams, bigK, binsize = binsize, maxbin = bigk_lmax, mask = mask_mod**2.)

    return bigK, mask_mod, el_, cl_big_k

################################################################################################################

def get_4pt(hmap, hmap2 = None, lmax_bigK=1000):
    if hmap2 is None:
        hmap2 = np.copy(hmap)
    bigK = hmap * hmap2
    if lmax_bigK == -1:
        return bigK
    else:        
        bigK_cl = H.anafast(bigK, lmax = lmax_bigK)
        
        return bigK, bigK_cl        

################################################################################################################

def get_median(els, cls, el_min, el_max):
    inds = np.where( (els>=el_min) & (els<=el_max))
    return np.median(cls[inds])

################################################################################################################
