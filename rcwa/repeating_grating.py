from rcwa import Layer, Slicer, Crystal, Grating, Material
import numpy as np
from typing import Union, Tuple
from numpy.typing import ArrayLike


class RepeatingGrating(Layer):
    """
    Base class that doesn't do much at all.
    """
    def __init__(self, er: complex = 1.0, ur: complex = 1.0, thickness: complex = 0.0, n: Union[complex, None] = None,
                 material: Union[None, Material] = None, crystal: Union[None, Crystal] = None):
        super().__init__(er=er, ur=ur, thickness=thickness, n=n, material=material, crystal=crystal)

    def _set_eun(self, n: float, n_void: float, er: float, er_void: float, ur: float, ur_void: float):
        if n is not None:
            self._er = np.square(n) # relative permittivity = refractive index**2
            self._er_void = np.square(n_void)
            self._ur = 1
            self._ur_void = 1
            self._n = n
            self._n_void = n_void
        else:
            self._er = er
            self._er_void = er_void
            self._ur = ur
            self._ur_void = ur_void
            self._n = np.sqrt(er*ur)
            self._n_void = np.sqrt(er_void*ur_void)

    def set_lv_period(self, tot_period, lattice_vector):
        if lattice_vector is not None:
            self.period = np.linalg.norm(lattice_vector)
            self.lattice_vector = lattice_vector
        else:
            self.lattice_vector = np.array([tot_period, 0])
            self.tot_period = tot_period

class RectangularRepeatingGrating(RepeatingGrating):
    """
    Class used for simple generation of 1D gratings. By default oriented with periodicity along the x-direction.

    :param single_period: Spatial period of each component of the grating
    :param num_period: Number of repeating components of the grating
    :param thickness: Thickness of the grating along z
    :param er: Permittivity of the grating material
    :param ur: Permeability of the grating material
    :param n: Refractive index of the grating material. Overrides permittivity/permeability if used.
    :param er_void: Permittivity of the voids in the grating
    :param ur_void: Permeability of the voids in the grating
    :param n_void: Refractive index of the voids in the grating. Overrides permittivity / permeability iif used.
    :param groove_width: Width of the empty spaces (voids) in the grating. Must be smaller than period
    :param nx: Number of points along x to divide the grating into
    :param lattice_vector: Explicit lattice vector for grating. Overrides period.
    """
    def __init__(self, single_period: float = 1, num_period: int = 1, er: float = 2, ur: float = 1, n: Union[float, None] = None,
                 thickness: float = 0.1, er_void: float = 1, ur_void: float = 1, n_void: float = 1,
                 groove_width: float = 0.5, nx: int = 500, lattice_vector: Union[None, ArrayLike] = None, offset: float = 0):

        if groove_width > single_period:
            raise ValueError(f'Groove width {groove_width} must be smaller than single period {single_period}')

        self.thickness = thickness
        self.nx = nx

        tot_period = single_period*num_period
        self._set_eun(n=n, n_void=n_void, er=er, er_void=er_void, ur=ur, ur_void=ur_void)
        self.set_lv_period(tot_period=tot_period, lattice_vector=lattice_vector)

        groove_fraction = groove_width / single_period
        offset_fraction = offset / single_period

        er_data, ur_data = self._er_data(
            er=er, ur=ur, n=n, er_void=er_void, ur_void=ur_void, n_void=n_void,
            Nx=nx, num_period=num_period, groove_fraction=groove_fraction, offset_fraction=offset_fraction)

        crystal = Crystal(self.lattice_vector, er=er_data, ur=ur_data)
        super().__init__(thickness=thickness, crystal=crystal)

    def _er_data(self, er: float = 2, er_void: float = 1, ur: float = 1, ur_void: float = 1,
                 n: Union[None, float] = None, n_void: float = 1, Nx: int = 500, num_period: int = 1,
                 groove_fraction: float = 0.5, offset_fraction: float = 0) -> Tuple[ArrayLike, ArrayLike]:
        '''
        Method used to define the grating geometry in the form of er, ur varying with position
        '''
        if n is not None:
            er_data = self._er_data_multiple(np.square(n_void), np.square(n), Nx, num_period, groove_fraction, offset_fraction)
            ur_data = np.ones(er_data.shape)
        else:
            er_data = self._er_data_multiple(er_void, er, Nx, num_period, groove_fraction, offset_fraction)
            ur_data = self._er_data_multiple(ur_void, ur, Nx, num_period, groove_fraction, offset_fraction)

        return er_data, ur_data

    def _er_data_single(self, val1: float, val2: float, Nx_each_period: int, switch_fraction: float, offset_fraction) -> ArrayLike:
        '''
        Returns an array defining permittivtiy/ permeability in one single grating component
        '''
        positions = np.linspace(1/Nx_each_period, 1, Nx_each_period)
        if offset_fraction >= 0:
            void_positions = np.logical_or(switch_fraction + offset_fraction <= positions, positions <= offset_fraction)
        else:
            void_positions = np.logical_and(switch_fraction + offset_fraction <= positions, positions <= offset_fraction % 1.0)
        return (val1 - val2) * void_positions + val2
    
    def _er_data_multiple(self, val1: float, val2: float, Nx: int, num_period: int, switch_fraction: float, offset_fraction) -> ArrayLike:
        Nx_each_period = int(Nx/num_period)
        er_ur_data = np.empty(Nx)
        for p in range(num_period - 1): # takes care of period #1 to period #num_period - 1
            er_ur_data[p*Nx_each_period:(p+1)*Nx_each_period] = self._er_data_single(val1, val2, Nx_each_period, switch_fraction, offset_fraction)
        # takes care of final period + some rounding error in the very end
        er_ur_data[(p+1)*Nx_each_period:] = self._er_data_single(val1, val2, Nx - (p+1)*Nx_each_period, switch_fraction, offset_fraction)
        return er_ur_data
