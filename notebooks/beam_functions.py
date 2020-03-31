"""


"""

import numpy as np

"""
Constants

"""
c = 3 * 10**8             # [m/s] speed of light
eps0 = 8.854 * 10**-12    # [F/m] permitivity of free space
mu0 = 4*np.pi * 10**-7    # [H/m] permeability of free space
lamb355 = 355 * 10**-9  # [m]
lamb118 = lamb355/3
h_planck = 6.626 * 10**-34 # [J s]
k118 = 2*np.pi/(118*10**(-9))
k88 = 2*np.pi/(88*10**(-9))
torr = 3.5*10**16

"""
Utility Functions

"""

class Params(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def norm(item):
    return np.array(item)/np.max(item)


"""
Derived Quantities

"""

def beam_radius(z, params):
    """
    Return the radius of a Gaussian beam a distance z from the focus.
    Parameters:
        z       - Required : distance from beam focus (float) [m]
        params  - Required : parameter list (Dict[str:float])
            "omega0" : beam waist at focus (float) [m]
            "zR" : rayleigh range of the gaussian beam (float) [m]

    Returns:
        radius of the gaussian beam at distance z (float) [m]

    """

    omega0 = params.omega0
    zR = params.zR
    # [m] * sqrt(1 + ([m]/[m])^2) = [m]
    return omega0 * np.sqrt(1+(z/zR)**2)

def peak_intensity_355(params):
    """
    Return the peak intensity of the 355 nm pulse.
    Parameters:
        params  - Required : parameter list (Dict[str:float])
            "omega0" : beam waist at focus (float) [m]
            "energy" : energy in one pulse of the laser (float) [J]
            "duration" : FWHM length of a laser pulse (float) [s]

    Returns:
        peak irradiance of the 355 pump beam (float) [J/(m^2 s), W/m^2]

    """

    omega0 = params.omega0
    energy = params.beam_energy
    duration = params.duration
    # [J] / (([m]^2) * [s]) = [W/m^2]
    return energy / ((np.pi * omega0**2) * duration)

def peak_amplitude_355(params):
    """
    Return the peak electric field amplitude of the 355 nm pulse.
    Parameters:
        params  - Required : parameter list (Dict[str:float])
            "omega0" : beam waist at focus (float) [m]
            "energy" : energy in one pulse of the laser (float) [J]
            "duration" : FWHM length of a laser pulse (float) [s]

    Returns:

    """

    return np.sqrt(2*peak_intensity_355(params) / (c * eps0))

def amplitude_355(z, params):
    """
    Return the profile of the 355 nm beam amplitude as a function of z.
    Parameters:


    Returns:

    """
    omega0 = params.omega0
    return (peak_amplitude_355(params) * (np.pi * omega0**2) /
            (np.pi * beam_radius(z, params)**2))


def normal_dist_2d(mu, sigma, x, y):
    """


    """
    mu0 = mu[0]
    mu1 = mu[1]
    sig0 = sigma[0]
    sig1 = sigma[1]
    z = (x-mu0)**2/sig0**2 + (y-mu1)**2/sig1**2
    return (1/(np.pi*sig0*sig1)
           * np.exp(-z/2))

def normalish_dist_2d(mu, sigma, x, y, exp=2):
    """


    """
    x0 = mu[0]
    y0 = mu[1]
    r = np.sqrt((x-x0)**2+(y-y0)**2)/sigma
    return np.exp(-r**exp/2)

def scale_beam_power(beam_profile, beam, params):
    """


    """
    dx = beam.dx
    power_factor = np.sqrt(2*params.beam_energy/(c*eps0*params.duration))
    beam_profile = beam_profile/np.sqrt(np.sum(beam_profile**2*dx**2)) * power_factor
    return beam_profile
