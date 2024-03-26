from django.db import models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
plt.ion()

class DiffractionGrating:
    

    """ Represents a diffraction grating for simulating potentials in electron microscopy. 

    -----

    Parameters

    size: int

        This is the size in pixels of the diffraction grating.

    absorption: float

        This parameter represents the absorption coefficient of the material forming the grating. 
        For an imaginary potential, it defines the strength of absorption. For a real potential, 
        it represents the potential strength.

    y: int

        The vertical position of the grating on the grid.

    gap_num: int

        The number of gaps in the diffraction grating.

    gap_size: int

        The size of each gap in the diffraction grating.

    gap_space: int

        The spacing between each gap in the diffraction grating.

    potential_type: str

        Specifies the type of potential, which can be either 'real' or 'imaginary'. 
        - 'real': Represents a real potential barrier.
        - 'imaginary': Represents an imaginary potential barrier, typically used to model 
          absorptive regions."""


    def __init__(self, size, absorption, y, gap_num, gap_size, gap_space, potential_type):

        self.size = size
        self.absorption = absorption
        self.y = y
        self.gap_num = gap_num
        self.gap_size = gap_size
        self.gap_space = gap_space
        self.potential_type = potential_type
        self.potential = self.create_potential()

    def create_potential(self):
        """ Creates an absorptive barrier as an imaginary potential on the grid."""
        width = 10
        if self.potential_type == 'imaginary':
            array = np.zeros(self.size, dtype=complex)      
            array[self.y : self.y + width, :] = -1.0j * self.absorption

        elif self.potential_type == 'real':
            array = np.zeros(self.size)
            array[self.y : self.y + width, :] = -1.0 * self.absorption
        g = np.arange(self.gap_num)
        g = (g - np.mean(g)) * self.gap_space + self.size[1] / 2
        for a0 in range(self.gap_num):
            gap_start = int(np.ceil(g[a0] - self.gap_size / 2))
            gap_end = int(np.ceil(g[a0] + self.gap_size / 2))
            array[self.y : self.y + width, gap_start:gap_end] = 0.0

        return array

    def is_in_barrier_region(self, x, M):
        """ Define the condition for a point to be part of the absorptive barrier."""
        return x > M/4 and x < 3*M/4


class PlaneWave:

    """ Represents a plane wave for use in electron microscopy simulations. """
    def __init__(self, size, wavelength, phase):

        self.size = size
        self.wavelength = wavelength
        self.phase = phase
        self.psi = self.initialize_wave()

    def initialize_wave(self):

        """ Initializes the wave function as a plane wave. """
        return np.exp(-1j * self.phase) * np.ones(self.size)

    def initialize_probe_wave(self, kcut):

        """ Initializes a focused beam or 'probe' wave function. """
        kn = np.fft.fftfreq(self.size)
        aperture_function = np.where(np.abs(kn) < kcut, 1, 0)
        return np.fft.ifft(aperture_function * self.psi)

    def multislice(self, potential:DiffractionGrating):

        """ Implements the multislice algorithm. """
        dy=1.0
        sigma=1.0
        psi = self.psi
        M = psi.shape[0]
        N_slices = potential.potential.shape[0]
        results = [psi]
        kx = np.fft.fftfreq(M)

        for n in range(N_slices):
            t = transmission_function(potential.potential[n], sigma)
            psi = np.fft.ifft(np.fft.fft(psi * t) * fresnel_propagator(kx, self.wavelength, dy))
            results.append(psi)

        return results


class MultisliceResult:

    def __init__(self, results):
        self.results = results

    def show(self, display_complex="grayscale"):

        for i, result in enumerate(self.results):
            if display_complex == "grayscale":
                plt.figure()
                im = plt.imshow(np.abs(result)**2, cmap='gray')
                plt.colorbar(im, extend='both')
                plt.title(f"Result {i}")
                plt.show()

            elif display_complex == "domain_coloring":

                try:
                    color_image = domain_coloring(result)
                    plt.figure()
                    plt.imshow(color_image)
                    plt.title(f"Domain Coloring Result {i}")
                    plt.show()

                except Exception as e:

                    print(f"Error in show: {e}")


    def animate(self):
        fig, ax = plt.subplots()
        im = ax.imshow(np.abs(self.results[0])**2, cmap='gray', animated=True)

        def update(frame):
            im.set_array(np.abs(self.results[frame])**2)
            return im,

        ani = animation.FuncAnimation(fig, update, frames=len(self.results), blit=True, interval=100, repeat=True)
        plt.show(block=True)

def transmission_function(V, sigma):

    """ Calculates the transmission function. """
    return np.exp(1j * sigma * V)

def fresnel_propagator(kx, lambda_, dy):

    """ Calculates the Fresnel propagator. """
    return np.exp(-1j * np.pi * lambda_ * kx**2 * dy)

def visualize_results(results):

    """ Visualizes the results of the multislice simulation. """

    for i, result in enumerate(results):
        result_abs = np.abs(result)**2
        plt.figure()
        im = plt.imshow(result_abs, cmap='gray')
        plt.colorbar(im, extend='both')
        plt.title(f"Result {i}")
        plt.show()
        pass

def domain_coloring(z):

    """ Applies domain coloring to a complex array. """
    H = np.angle(z) / (2 * np.pi) + 0.5
    S = np.ones_like(H)
    V = np.abs(z) / (1 + np.abs(z))
    return hsv_to_rgb(np.dstack((H, S, V)))

def visualize_with_domain_coloring(results):

    for i, result in enumerate(results):
        try:
            color_image = domain_coloring(result)
            plt.figure()
            plt.imshow(color_image)
            plt.title(f"Domain Coloring Result {i}")
            pass

        except Exception as e:
            print(f"Error in visualize_with_domain_coloring: {e}")

    plt.show()
    pass

def mock (index):

    x = index*np.sin(index)
    y = index*np.cos(index)
    return np.array([x , y])

def animate_results(results):

    """ Creates an animation of the multislice simulation results. """
    fig, ax = plt.subplots() 
    result_abs = np.abs(results[0])**2
    im = ax.imshow(result_abs, cmap='gray', animated=True)

    def update(frame):
        """ Update function for animation. """
        result_abs = np.abs(results[frame])**2
        print(frame)
        print(result_abs)
        im = ax.imshow(result_abs, cmap='gray', animated=True)
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(results), blit=True, interval=100, repeat=True)

    plt.show(block=True)
    

def get_user_input():
    """ Prompts the user for simulation parameters with error handling. """
    while True:
        try:
            size = int(input("Enter grid size (e.g., 512): "))
            wavelength = float(input("Enter wavelength (e.g., 2.0): "))
            phase = float(input("Enter phase (e.g., 0.7854 for π/4): "))
            potential_type = input("Enter potential type ('real' or 'imaginary'): ")
            absorption = float(input("Enter absorption (for imaginary potential, e.g., 0.2): "))
            return size, wavelength, phase, potential_type, absorption
        except ValueError:
            print("Invalid input. Please enter the correct data types for each parameter.")


# Main execution

