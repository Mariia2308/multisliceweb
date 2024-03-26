import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb

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