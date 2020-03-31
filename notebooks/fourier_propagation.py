"""


"""
import pickle
import os
import csv
import datetime

import numpy as np
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import tqdm

from beam_functions import (
    beam_radius,
    normal_dist_2d,
    amplitude_355,
    normalish_dist_2d,
    scale_beam_power,
    )

# Change the way downsampling is done to make it easy to monkey-patch in another array (gain118, loss118)

def next_power_of_2(N):
    return np.power(2, int(np.ceil(np.log2(N))))


class FourierBeam3D:
    def __init__(self, lamb, x, y, z, downsample_rate):

        downsample_rate = list(downsample_rate)
        if len(downsample_rate) == 1:
            self.dsx = downsample_rate[0]
            self.dsy = downsample_rate[0]
            self.dsz = downsample_rate[0]
        if len(downsample_rate) == 2:
            self.dsx = downsample_rate[0]
            self.dsy = downsample_rate[0]
            self.dsz = downsample_rate[1]
        if len(downsample_rate) == 3:
            self.dsx = downsample_rate[0]
            self.dsy = downsample_rate[1]
            self.dsz = downsample_rate[2]

        self.lamb = lamb

        self.x = x
        self.y = y
        self.z = z

        self.Nx = len(x)
        self.Ny = len(y)
        self.Nz = len(z)

        self.Nx_ds = int(self.Nx/self.dsx)
        self.Ny_ds = int(self.Ny/self.dsy)
        self.Nz_ds = int(self.Nz/self.dsz)

        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0]

        self.dfx = 1/(self.Nx*self.dx)
        self.fx_fx = np.arange((-self.Nx/2)*self.dfx, (self.Nx/2)*self.dfx, self.dfx)
        self.fx_fx = np.expand_dims(self.fx_fx, axis=0)

        self.dfy = 1/(self.Ny*self.dy)
        self.fy_fy = np.arange((-self.Ny/2)*self.dfy, (self.Ny/2)*self.dfy, self.dfy)
        self.fy_fy = np.expand_dims(self.fy_fy, axis=1)

        self.fz_fxfy = np.sqrt((1/self.lamb)**2 - np.power(self.fx_fx,2) - np.power(self.fy_fy,2))

    def set_initial_E(self, E=None):

        if E is None:
            self.E = np.zeros((self.Nx, self.Ny), dtype=complex)
        else:
            self.E = E

        self.E_fx = np.zeros((self.Nx, self.Ny), dtype=complex)
        self.E_fx = ifftshift(fft2(fftshift(self.E)))

        self.E_xz = np.zeros((self.Nx_ds, self.Ny_ds, self.Nz_ds), dtype=complex)
        self.E_fxz = np.zeros((self.Nx_ds, self.Ny_ds, self.Nz_ds), dtype=complex)

        self.E_xz[:,:,0] = self.E[::self.dsx, ::self.dsy]
        self.E_fxz[:,:,0] = self.E_fx[::self.dsx, ::self.dsy]

        #print(f"Nx: {self.Nx}, x_x: {len(self.x)}, fx_fx: {self.fx_fx.shape}, fz_fxfy: {len(self.fz_fxfy)}")
        #print(f"E: {self.E.shape}, E_fx: {self.E_fx.shape}, E_xz: {self.E_xz.shape}, E_fxz: {self.E_fxz.shape}")

    def update_real(self, z_idx):
        self.E = fftshift(ifft2(ifftshift(self.E_fx)))
        self.downsample_save(z_idx, self.E, self.E_xz)

    def update_fourier(self, z_idx):
        self.E_fx = (ifftshift(fft2(fftshift(self.E)))
                               * np.exp(1j*2*np.pi*self.fz_fxfy*self.dz))
        self.downsample_save(z_idx, self.E_fx, self.E_fxz)

    def downsample_save(self, idx, array, save_array):
        if idx % self.dsz == 0:
            slot = int(idx/self.dsz)
            save_array[:, :, slot] = array[::self.dsx, ::self.dsy]

    def finalize(self):
        self.E = None
        self.E_fx = None
        self.fz_fxfy = None

        self.fields = {
            'field': (np.abs(self.E_xz), "Real Space Field, |E|: (V/m)"),
            'intensity': (np.abs(self.E_xz)**2, "Real Space Intensity, I: (V/m)^2")
        }

    def plot_2d_slice(self, ax, x=None, y=None, z=None, var='field', norm=False):
        nano = 10**-9
        micro = 10**-6
        milli = 10**-3
        degree = np.pi/180

        field, label = self.fields[var]

        if x is not None:
            dim1 = self.z[::self.dsz]/milli
            dim2 = self.y[::self.dsy]/micro
            dim1_label = "z-Distance (mm)"
            dim2_label = "y-Distance (um)"
            field = field[x,:,:]
        elif y is not None:
            dim1 = self.z[::self.dsz]/milli
            dim2 = self.x[::self.dsx]/micro
            dim1_label = "z-Distance (mm)"
            dim2_label = "x-Distance (um)"
            field = field[:,y,:]
        elif z is not None:
            dim1 = self.x[::self.dsx]/micro
            dim2 = self.y[::self.dsy]/micro
            dim1_label = "x-Distance (um)"
            dim2_label = "y-Distance (um)"
            field = field[:,:,z]

        if norm:
            field = field/np.amax(field)

        fig = plt.gcf()
        im = ax.contourf(dim1, dim2, field, 32)
        fig.colorbar(im, ax=ax)
        ax.set_title(label)
        ax.set_xlabel(dim1_label)
        ax.set_ylabel(dim2_label)

    def plot_1d_slice(self, ax, x=None, y=None, z=None, var='field', norm=False):
        nano = 10**-9
        micro = 10**-6
        milli = 10**-3
        degree = np.pi/180

        field, label = self.fields[var]

        if x is not None and y is not None:
            dim = self.z[::self.dsz]/milli
            dim_label = "z_distance (mm)"
            field = field[x,y,:]
        elif x is not None and z is not None:
            dim = self.y[::self.dsy]/micro
            dim_label = "y_distance (um)"
            field = field[x,:,z]
        elif y is not None and z is not None:
            dim = self.x[::self.dsx]/micro
            dim_label = "x_distance (um)"
            field = field[:,y,z]

        if norm:
            field = field/np.amax(field)

        ax.plot(dim, field)
        ax.set_title(label)
        ax.set_xlabel(dim_label)

def class_sizes(item):
    all_attrs = [attr for attr in dir(item) if not attr.startswith('__')]
    all_attrs = [attr for attr in all_attrs
        if not callable(getattr(item, attr))]
    sizes = [
            [attr, type(getattr(item, attr)),
             np.product(getattr(item, attr).shape)]
                if isinstance(getattr(item, attr), np.ndarray)
                else [attr, type(getattr(item, attr))]
                for attr in all_attrs
            ]
    return sizes

def save_beams(filename, beams, params):
    with open(filename, 'bx') as file:
        beam355, beam118 = beams
        pickle.dump((beam355, beam118, params), file)

def record_beams(record_filename, beam_filename, params):
    keys = []
    for key in params.__dict__:
        keys.append(key)
    keys = sorted(keys)
    keys.append('filename')

    if not os.path.exists(record_filename):
        with open(record_filename, 'x') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(keys)

    data = [getattr(params, key) for key in keys[:-1]] + [beam_filename]
    with open(record_filename, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data)

def load_beams(filename):
    with open(filename, 'rb') as file:
        (beam355, beam118, params) = pickle.load(file)
    return (beam355, beam118, params)

def fourier_update_loop(beam355, beam118, params, flag_118=True):
    """


    """

    k118 = 2*np.pi/(118*10**(-9))
    k88 = 2*np.pi/(88*10**(-9))

    chi2 = params.chi2
    chi3 = params.chi3
    PXe = params.PXe

    t = tqdm.trange(params.Nz, leave=True)

    for iz in t:

        # tqdm stuff, comment out if you don't want progress bar messiness
        t.set_description(
        f"Central Vals, 355: "
        f"{np.abs(beam355.E[int(beam355.Nx/2),int(beam355.Nx/2)]):.3E}, "
        f"118: {np.abs(beam118.E[int(beam118.Nx/2),int(beam118.Nx/2)]):.3E}")
        t.refresh() # to immediately show the update

        # Propagate the beams forward
        beam355.update_real(iz)
        beam355.update_fourier(iz)

        if flag_118:
            beam118.update_real(iz)

            unit_vec_118_direction = beam118.E[:,:] / np.abs(beam118.E[:,:])

            interp355 = interp2d(beam355.x, beam355.y, np.abs(beam355.E[:,:]))

            gain_mag = (chi3*PXe*k118
                    * np.abs(interp355(beam118.x, beam118.y))**3)
            gain118 = (np.ones(gain_mag.shape, dtype=complex)
                    * np.abs(gain_mag) * unit_vec_118_direction)

            # loss_mag = (chi2*PXe*k88
            #     * np.abs(interp355(beam118.x, beam118.y))
            #     * np.abs(beam118.E[:,:]))
            loss_mag = (chi2*PXe*PXe
                * np.abs(beam118.E[:,:]))
            loss118 = (np.ones(loss_mag.shape, dtype=complex)
                * np.abs(loss_mag) * unit_vec_118_direction)

            beam118.E[:,:] += (gain118 - loss118)*beam118.dz
            beam118.update_fourier(iz)
            beam118.downsample_save(iz, gain118, beam118.gain)
            beam118.downsample_save(iz, loss118, beam118.loss)

    # finalize beams and monkey-patch in 118 fields of interest
    beam355.finalize()
    beam118.finalize()
    beam118.fields['gain'] = (np.abs(beam118.gain),
                            "Real Space 118 Gain, |E|: (V/m)")
    beam118.fields['loss'] = (np.abs(beam118.loss),
                            "Real Space 118 Loss, |E|: (V/m)")

    return (beam355, beam118)

def process_beam118(beam118):
    """

    """
    output = np.sum(np.abs(beam118.E_xz[:,:,-1])**2)
    loss = np.sum(np.abs(beam118.loss)**2)
    return(output, loss)

def derive_convenience_params(params):
    params.zR = params.b / 2  # [m] Rayleigh range
    params.omega0 = np.sqrt(params.lamb355 * params.zR/ np.pi) # [m] beam radius at the focus
    params.f = params.z_max/2 # focal length of the 355 beam
    params.width = beam_radius(params.f, params) # [m] standard-dev width of the beam at the front plane of sim


def scan_parameter(params,
                   scan_param,
                   scan_param_values,
                   directory):

    for i,value in enumerate(scan_param_values):
        print(f"Scan {i+1}/{len(scan_param_values)}. Scan param: {scan_param} = {value}")
        setattr(params, scan_param, value)
        derive_convenience_params(params)

        # Grid Parameters:
        z = np.linspace(-params.z_max/2, params.z_max/2, params.Nz)
        dz = z[1] - z[0]

        dx355 = params.lamb355/np.sqrt(1.99999)
        Nx355 = next_power_of_2(int(2*params.x_max/params.lamb355))
        x355 = np.arange(-Nx355/2, Nx355/2)*dx355

        dx118 = params.lamb118/np.sqrt(1.99999)
        Nx118 = next_power_of_2(int(2*params.x_max/params.lamb118/3)) * 3
        x118 = np.arange(-Nx118/2, Nx118/2)*dx118

        # create 355 beam
        beam355 = FourierBeam3D(params.lamb355,
                                x355,
                                x355,
                                z,
                                downsample_rate=(params.dsx,
                                                 params.dsy,
                                                 params.dsz)
                                )
        x_grid, y_grid = np.meshgrid(x355, x355)
        """
        E355_init = (normal_dist_2d((0,0),
                                    (params.width, params.width),
                                    x_grid,
                                    y_grid))
        E355_init = E355_init/np.amax(E355_init) * amplitude_355(0, params)
        E355_init = (E355_init *
                    (np.exp( -1j * (np.power(x_grid,2) + np.power(y_grid,2))
                            / (2*params.f) * 2*np.pi/params.lamb355) ))
        """
        E355_init = normalish_dist_2d((0,0),
                                       params.width,
                                       x_grid,
                                       y_grid,
                                       exp=params.exponent)
        E355_init = scale_beam_power(E355_init, beam355, params)
        E355_init = (E355_init *
                    (np.exp( -1j * (np.power(x_grid,2) + np.power(y_grid,2))
                            / (2*params.f) * 2*np.pi/params.lamb355) ))
        beam355.set_initial_E(E355_init)

        # create 118 beam
        beam118 = FourierBeam3D(params.lamb118,
                                x118,
                                x118,
                                z,
                                downsample_rate=(params.dsx*3,
                                                 params.dsy*3,
                                                 params.dsz)
                                )
        E118_init = np.ones((Nx118, Nx118), dtype=complex)*10**-10
        beam118.set_initial_E(E118_init)
        beam118.gain = np.zeros(beam118.E_xz.shape, dtype=complex)
        beam118.loss = np.zeros(beam118.E_xz.shape, dtype=complex)

        # simulate beams
        beam355, beam118 = fourier_update_loop(beam355,
                                               beam118,
                                               params)
        # save beams
        timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        beam_filename = (directory
                + "parameter_scan_"
                + scan_param
                + '_'
                + timestamp
                + ".beam")
        record_filename = (directory
                + "parameter_scan_"
                + scan_param
                + ".csv")
        save_beams(beam_filename, (beam355, beam118), params)
        record_beams(record_filename, beam_filename, params)
