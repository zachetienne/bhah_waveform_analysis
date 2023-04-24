import numpy as np
import re
from scipy.optimize import curve_fit

def read_psi4(file_name):
    """
    Reads an ASCII file with a header describing the real and imaginary parts of the data for each mode.
    Returns the data in a convenient format to access the real and imaginary parts given l, m values.

    Args:
        file_name (str): The name of the file to read.

    Returns:
        tuple: A tuple containing the time numpy array and a dictionary with keys (l, m) containing the data.
    """
    mode_data = {}
    time_data = []

    with open(file_name, 'r') as file:
        # Read the lines and ignore lines starting with #
        lines = [line for line in file.readlines() if not line.startswith('#')]

    # Convert lines to arrays and sort by time
    data = np.array([list(map(np.float64, line.split())) for line in lines])
    data = data[np.argsort(data[:, 0])]

    # Remove duplicate times
    _, index = np.unique(data[:, 0], return_index=True)
    data = data[index]

    # Store time data
    time_data = data[:, 0]

    # Extract l from the filename
    l = int(file_name.split('_')[1][1:].split('-')[0])

    # Loop through columns and store real and imaginary parts in mode_data
    for m in range(-l, l + 1):
        idx = 1 + 2 * (m + l)  # Calculate the index of the real part
        mode_data[(l, m)] = data[:, idx], data[:, idx + 1]

    return time_data, mode_data


def compute_first_derivative(time, data):
    """
    Calculates the time derivative of the input data using a second-order finite difference stencil.

    Args:
        time (numpy.ndarray): A numpy array containing time values.
        data (numpy.ndarray): A numpy array containing the data to be differentiated.

    Returns:
        numpy.ndarray: A numpy array containing the time derivative of the input data.
    """
    dt = time[1] - time[0]
    derivative = np.zeros_like(data)
    # Second-order in the interior:
    derivative[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
    # Drop to first-order at the endpoints
    derivative[0] = (data[1] - data[0]) / dt
    derivative[-1] = (data[-1] - data[-2]) / dt

    return derivative

def compute_second_derivative(time, data):
    """
    Computes the second time derivative of the input data using the second-order finite difference method,
    with upwind/downwind stencils for the endpoints.

    Args:
        time (numpy.ndarray): A numpy array containing time values.
        data (numpy.ndarray): A numpy array containing data for which the second time derivative is to be calculated.

    Returns:
        numpy.ndarray: A numpy array containing the second time derivative of the input data.
    """
    dt = time[1] - time[0]
    n = len(data)
    second_derivative = np.zeros(n)

    # Interior points using central finite difference
    second_derivative[1:-1] = (data[:-2] - 2 * data[1:-1] + data[2:]) / (dt ** 2)

    # Endpoint 0: forward finite difference (downwind)
    second_derivative[0] = (2 * data[0] - 5 * data[1] + 4 * data[2] - data[3]) / (dt ** 2)

    # Endpoint n-1: backward finite difference (upwind)
    second_derivative[-1] = (2 * data[-1] - 5 * data[-2] + 4 * data[-3] - data[-4]) / (dt ** 2)

    return second_derivative


def process_wave_data(time, real, imag):
    """
    Calculates the cumulative phase and amplitude of a gravitational wave signal.

    Args:
        time (numpy.ndarray): A numpy array containing time values.
        real (numpy.ndarray): A numpy array containing the real part of the signal.
        imag (numpy.ndarray): A numpy array containing the imaginary part of the signal.

    Returns:
        tuple: A tuple containing three numpy arrays (time, cumulative_phase, amplitude).
    """
    # Calculate the amplitude of the gravitational wave signal.
    amplitude = np.sqrt((real**2 + imag**2), dtype=np.float64)

    # Calculate the instantaneous phase of the gravitational wave signal.
    phase = np.arctan2(imag, real, dtype=np.float64)

    # Initialize a variable to count the number of full cycles completed by the signal.
    cycles = 0

    # Initialize an empty list to store the cumulative phase of the signal.
    cum_phase = []

    # Set the variable `last_phase` to the first value of the instantaneous phase array.
    last_phase = phase[0]

    # Iterate over each value of the instantaneous phase array.
    for ph in phase:
        # Check if the absolute difference between the current phase and the previous phase
        # is greater than or equal to pi (to identify phase wrapping).
        if np.abs(ph - last_phase) >= np.pi:
            # If the current phase is positive, the phase wrapped from a positive value
            # to a negative value, so decrease the `cycles` variable by 1.
            if ph > 0:
                cycles -= 1
            # If the current phase is negative, the phase wrapped from a negative value
            # to a positive value, so increase the `cycles` variable by 1.
            if ph < 0:
                cycles += 1

        # Calculate the cumulative phase for the current time step and append it to the `cum_phase` list.
        cum_phase.append(ph + 2 * np.pi * cycles)

        # Update the `last_phase` variable with the current phase value.
        last_phase = ph

    # Calculate the cumulative phase for the current time step and append it to the `cum_phase` list.
    cum_phase = np.array(cum_phase, dtype=np.float64)

    # Compute the time derivative of the cumulative phase using a second-order finite difference stencil.
    cum_phase_derivative = compute_first_derivative(time, cum_phase)

    return time, cum_phase, amplitude, cum_phase_derivative

import sys

def fit_quadratic_and_output_min_omega(time, omega):
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    # Filter the data for t=100 to t=300
    time_filtered = time[(time >= 100) & (time <= 300)]
    omega_filtered = omega[(time >= 100) & (time <= 300)]

    # Fit a quadratic curve to the Omega data using nonlinear least squares
    params, _ = curve_fit(quadratic, time_filtered, omega_filtered)

    # Find the extremum value of the quadratic curve
    a, b, c = params
    extremum_x = -b / (2 * a)
    omega_min_quad_fit = np.fabs(quadratic(extremum_x, a, b, c))
    omega_at_t_zero = np.fabs(quadratic(0.0, a, b, c))

    print(f"The extremum of the quadratic curve occurs at t = {extremum_x:.15f} with omega = {omega_min_quad_fit:.15f} . implied omega(t=0) = {omega_at_t_zero:.15f}")
    return np.fabs(omega_at_t_zero)
    
import numpy as np

def perform_complex_fft(time, real, imag):
    """
    Performs a complex Fast Fourier Transform (FFT) on the input time, real, and imaginary data.

    Args:
        time (numpy.ndarray): A numpy array containing time values.
        real (numpy.ndarray): A numpy array containing the real part of the signal.
        imag (numpy.ndarray): A numpy array containing the imaginary part of the signal.

    Returns:
        tuple: A tuple containing two numpy arrays (frequencies, fft_data).
    """
    # Combine the real and imaginary data into a single complex signal
    complex_signal = real + 1j * imag

    # Perform the complex FFT
    fft_data = np.fft.fft(complex_signal)

    # Calculate the frequency values
    dt = time[1] - time[0]
    n = len(time)
    frequencies = np.fft.fftfreq(n, d=dt)

    return frequencies, fft_data

    
def main():
    """
    Main function that reads the gravitational wave data file, processes the data,
    and saves the output to a file. The input filename is provided via command line.
    """
    if len(sys.argv) != 2:
        print("Usage: python3 phase_amp_omega.py <gravitational_wave_data.asc or .txt>")
        sys.exit(1)

    file_name = sys.argv[1]

    if not file_name.endswith('.asc') and not file_name.endswith('.txt'):
        print("Error: Input file must have a '.asc' or '.txt' extension.")
        sys.exit(1)

    time, mode_data = read_psi4(file_name)
    real, imag = mode_data[(2, 2)]
    print(real, imag)

    time, cumulative_phase, amplitude, omega = process_wave_data(time, real, imag)

    
    output_file = file_name
    if file_name.endswith('.asc'):
        output_file = output_file.replace('.asc', '_phase_amp_omega.asc')
    elif file_name.endswith('.txt'):
        output_file = output_file.replace('.txt', '_phase_amp_omega.txt')

    with open(output_file, 'w') as file:
        file.write("# Time    cumulative_phase    amplitude    omega\n")
        for t, cp, a, o in zip(time, cumulative_phase, amplitude, omega):
            file.write(f"{t:.15f} {cp:.15f} {a:.15f} {o:.15f}\n")

    print(f"Processed data has been saved to {output_file}")

    min_omega = fit_quadratic_and_output_min_omega(time, omega)

    # Perform the FFT
    fft_result = np.fft.fft(real + 1j * imag)

    # Calculate angular frequencies
    omega_list = np.fft.fftfreq(len(time), time[1] - time[0]) * 2 * np.pi

    # Just below Eq. 27 in https://arxiv.org/abs/1006.1632
    for i, omega in enumerate(omega_list):
        if np.fabs(omega) <= min_omega:
            fft_result[i] *= 1 / (1j * min_omega) ** 2
        else:
            fft_result[i] *= 1 / (1j * np.fabs(omega)) ** 2

    # Now perform the inverse FFT
    second_integral_complex = np.fft.ifft(fft_result)

    # Separate the real and imaginary parts of the second time integral
    second_integral_real = np.real(second_integral_complex)
    second_integral_imag = np.imag(second_integral_complex)
    
    # Save the output to a file with _strain.{asc or txt} extension
    output_file = file_name
    if file_name.endswith('.asc'):
        output_file = output_file.replace('.asc', '_strain.asc')
    elif file_name.endswith('.txt'):
        output_file = output_file.replace('.txt', '_strain.txt')

    with open(output_file, 'w') as file:
        file.write("# Time    Second_Integral_Real    Second_Integral_Imag\n")
        for t, real, imag in zip(time, second_integral_real, second_integral_imag):
            file.write(f"{t:.15f} {real:.15f} {imag:.15f}\n")

    print(f"Second time integral data has been saved to {output_file}")

    # Calculate the second time derivative of second_integral_real and second_integral_imag
    second_derivative_real = compute_second_derivative(time, second_integral_real)
    second_derivative_imag = compute_second_derivative(time, second_integral_imag)
    with open("check.txt", 'w') as file:
        file.write("# Time    Second_Integral_Real    Second_Integral_Imag\n")
        for t, real, imag in zip(time, second_derivative_real, second_derivative_imag):
            file.write(f"{t:.15f} {real:.15f} {imag:.15f}\n")
    

    
if __name__ == "__main__":
    main()
