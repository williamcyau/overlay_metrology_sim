from rcwa import Layer, Slicer, Crystal, Grating
import numpy as np
from typing import Union, Tuple
from numpy.typing import ArrayLike


class NonPeriodicGrating(Layer):
    """
    Base class that doesn't do much at all.
    """

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

class RectangularNonperiodicGrating(NonPeriodicGrating):
    """
    Class used for simple generation of 1D gratings. By default oriented with periodicity along the x-direction.

    :param tot_period: total spatial length of the grating
    :param left_bounds: Spatial locations of the left bounds of each component of the grating
    :param right_bounds: Spatial locations of the right bounds of each component of the grating
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
    def __init__(self, tot_period: float = 1, left_bounds: Union[None, ArrayLike] = None, right_bounds: Union[None, ArrayLike] = None, er: float = 2, ur: float = 1, n: Union[float, None] = None,
                 thickness: float = 0.1, er_void: float = 1, ur_void: float = 1, n_void: float = 1,
                 nx: int = 500, lattice_vector: Union[None, ArrayLike] = None):

        self.thickness = thickness
        self.nx = nx

        self._set_eun(n=n, n_void=n_void, er=er, er_void=er_void, ur=ur, ur_void=ur_void)
        self.set_lv_period(tot_period=tot_period, lattice_vector=lattice_vector)

        assert len(left_bounds) == len(right_bounds), f"left and right bounds must be of the same length, got left {len(left_bounds)} and right {len(right_bounds)}"
        # assert np.all(left_bounds_fraction < right_bounds_fraction), "right bounds must be larger than left bounds"
        
        left_bounds_fraction = left_bounds / tot_period
        right_bounds_fraction = right_bounds / tot_period
        
        er_data, ur_data = self._er_data(
            er=er, ur=ur, n=n, er_void=er_void, ur_void=ur_void, n_void=n_void,
            Nx=nx, left_bounds_fraction=left_bounds_fraction, right_bounds_fraction=right_bounds_fraction)

        crystal = Crystal(self.lattice_vector, er=er_data, ur=ur_data)
        super().__init__(thickness=thickness, crystal=crystal)

    def _er_data(self, er: float = 2, er_void: float = 1, ur: float = 1, ur_void: float = 1,
                 n: Union[None, float] = None, n_void: float = 1, Nx: int = 500, 
                 left_bounds_fraction: Union[None, ArrayLike] = None, right_bounds_fraction: Union[None, ArrayLike] = None) -> Tuple[ArrayLike, ArrayLike]:
        '''
        Method used to define the grating geometry in the form of er, ur varying with position
        '''
        if n is not None:
            er_data = self._erur_fill_entire(np.square(n_void), np.square(n), Nx, left_bounds_fraction=left_bounds_fraction, right_bounds_fraction=right_bounds_fraction)
            ur_data = np.ones(er_data.shape)
        else:
            er_data = self._erur_fill_entire(er_void, er, Nx, left_bounds_fraction=left_bounds_fraction, right_bounds_fraction=right_bounds_fraction)
            ur_data = self._erur_fill_entire(ur_void, ur, Nx, left_bounds_fraction=left_bounds_fraction, right_bounds_fraction=right_bounds_fraction)

        return er_data, ur_data
    
    def _erur_fill_entire(self, val1: float, val2: float, Nx: int, left_bounds_fraction: Union[None, ArrayLike] = None, right_bounds_fraction: Union[None, ArrayLike] = None) -> ArrayLike:
        positions = np.linspace(1/Nx, 1, Nx)
        er_ur_data = np.full_like(positions, val1)
        
        for i in range(len(left_bounds_fraction)):
            val2_positions = np.logical_and(left_bounds_fraction[i] <= positions, positions <= right_bounds_fraction[i])
            er_ur_data[val2_positions] = val2
        return er_ur_data
    
    # def _erur_fill_entire(self, val1: float, val2: float, Nx: int, left_bounds_fraction: Union[None, ArrayLike] = None, right_bounds_fraction: Union[None, ArrayLike] = None) -> ArrayLike:
    #     positions = np.linspace(1/Nx, 1, Nx)
    #     er_ur_data = np.full_like(positions, val1)
        
    #     for i in range(len(left_bounds_fraction)):
    #         val2_positions = np.logical_and(left_bounds_fraction[i] < positions, positions < right_bounds_fraction[i])
    #         er_ur_data[val2_positions] = val2
    #     return er_ur_data
