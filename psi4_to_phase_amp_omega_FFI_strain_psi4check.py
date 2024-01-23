"""
This module processes gravitational wave data from numerical relativity simulations.
It provides functions to read waveform data files, compute derivatives, process wave data,
perform complex Fast Fourier Transforms (FFT), and fit data to a quadratic function.
The main functionality involves reading gravitational wave data, extracting relevant
information like the phase and amplitude, and performing analysis like FFT and
quadratic fitting to extract physical properties from the waveforms.
It's designed to work with ASCII files containing gravitational wave data from simulations.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""
import sys
from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit  # type: ignore


def read_BHaH_psi4_file(
    file_name: str,
) -> Tuple[
    float,
    NDArray[np.float64],
    Dict[Tuple[int, int], Tuple[NDArray[np.float64], NDArray[np.float64]]],
]:
    """
    Read an ASCII file with a header describing the real and imaginary parts of the data for each mode.
    Return the data in a format to access the real and imaginary parts given l, m values.

    :param file_name: The name of the file to read.
    :return: A tuple containing the time numpy array and a dictionary with keys (l, m) containing the data.
    """

    def extract_radius_from_filename(filename: str) -> float:
        """
        Extracts the radius value from a given filename.

        :param filename: The filename in the format 'Rpsi4_lX-rYYYY.Z.txt'
        :return: The extracted radius value as a float.

        :raises ValueError: If the filename format is incorrect.
        """

        try:
            # Split the filename on '-r' and take the second part
            # Then split on '.' and take the first part to get the radius value
            radius_str = filename.split("-r")[1].split(".")[0]

            # Convert the radius string to float
            radius = float(radius_str)

            return radius
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid filename format.") from exc

    mode_data = {}

    with open(file_name, mode="r", encoding="utf-8") as file:
        # Read the lines and ignore lines starting with #
        lines = [line for line in file.readlines() if not line.startswith("#")]

    # Convert lines to arrays and sort by time
    data = np.array([list(map(np.float64, line.split())) for line in lines])
    data = data[np.argsort(data[:, 0])]

    # Remove duplicate times
    _, index = np.unique(data[:, 0], return_index=True)
    data = data[index]

    # Store time data
    time_data = data[:, 0]

    # Extract l from the filename
    l = int(file_name.split("_")[1][1:].split("-")[0])

    # Loop through columns and store real and imaginary parts in mode_data
    for m in range(-l, l + 1):
        idx = 1 + 2 * (m + l)  # Calculate the index of the real part
        mode_data[(l, m)] = data[:, idx], data[:, idx + 1]

    return extract_radius_from_filename(file_name), time_data, mode_data


def compute_first_derivative_in_time(
    time: NDArray[np.float64], data: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate the time derivative of the input data using a second-order finite difference stencil.

    :param time: A numpy array containing time values.
    :param data: A numpy array containing the data to be differentiated.
    :return: A numpy array containing the time derivative of the input data.

    >>> time = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    >>> data = np.array([0, 1, 4, 9, 16], dtype=np.float64)
    >>> compute_first_derivative_in_time(time, data)
    array([1., 2., 4., 6., 7.])
    """
    dt = time[1] - time[0]
    derivative = np.zeros_like(data)
    # Second-order in the interior:
    derivative[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
    # Drop to first-order at the endpoints
    derivative[0] = (data[1] - data[0]) / dt
    derivative[-1] = (data[-1] - data[-2]) / dt

    return derivative


def compute_second_derivative_in_time(
    time: NDArray[np.float64], data: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the second time derivative of the input data using the second-order finite difference method,
    with upwind/downwind stencils for the endpoints.

    :param time: A numpy array containing time values.
    :param data: A numpy array containing data for which the second time derivative is to be calculated.
    :return: A numpy array containing the second time derivative of the input data.

    >>> time = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    >>> data = np.array([0, 1, 4, 9, 16], dtype=np.float64)
    >>> compute_second_derivative_in_time(time, data)
    array([2., 2., 2., 2., 2.])
    """
    dt = time[1] - time[0]
    n = len(data)
    second_derivative = np.zeros(n)

    # Interior points using central finite difference
    second_derivative[1:-1] = (data[:-2] - 2 * data[1:-1] + data[2:]) / (dt**2)

    # Endpoint 0: forward finite difference (downwind)
    second_derivative[0] = (2 * data[0] - 5 * data[1] + 4 * data[2] - data[3]) / (
        dt**2
    )

    # Endpoint n-1: backward finite difference (upwind)
    second_derivative[-1] = (2 * data[-1] - 5 * data[-2] + 4 * data[-3] - data[-4]) / (
        dt**2
    )

    return second_derivative


def compute_psi4_wave_phase_and_amplitude(
    time: NDArray[np.float64], real: NDArray[np.float64], imag: NDArray[np.float64]
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Calculates the cumulative phase and amplitude of a gravitational wave signal.

    :param time: A numpy array containing time values.
    :param real: A numpy array containing the real part of the signal.
    :param imag: A numpy array containing the imaginary part of the signal.

    :return: A tuple containing four numpy arrays (time, cumulative_phase, amplitude, cumulative_phase_derivative).

    :raises ValueError: If the lengths of time, real, and imag arrays are not equal.
    """

    if not len(time) == len(real) == len(imag):
        raise ValueError("The lengths of time, real, and imag arrays must be equal.")

    # Calculate the amplitude of the gravitational wave signal.
    amplitude = np.sqrt(real**2 + imag**2)

    # Calculate the instantaneous phase of the gravitational wave signal.
    phase = np.arctan2(imag, real)

    # Initialize a variable to count the number of full cycles completed by the signal.
    cycles = 0

    # Initialize an empty list to store the cumulative phase of the signal.
    cum_phase = np.empty_like(time)  # Initialize cum_phase as a numpy array

    # Set the variable `last_phase` to the first value of the instantaneous phase array.
    last_phase = phase[0]

    # Iterate over each value of the instantaneous phase array.
    for i, ph in enumerate(phase):
        # Check if the absolute difference between the current phase and the previous phase
        # is greater than or equal to pi (to identify phase wrapping).
        if np.abs(ph - last_phase) >= np.pi:
            # Adjust the `cycles` variable based on the direction of phase wrapping.
            cycles += -1 if ph > 0 else 1

        # Calculate the cumulative phase for the current time step and append it to the `cum_phase` list.
        cum_phase[i] = ph + 2 * np.pi * cycles

        # Update the `last_phase` variable with the current phase value.
        last_phase = ph

    # Convert the cumulative phase list to a numpy array.
    cum_phase = np.array(cum_phase)

    # Compute the time derivative of the cumulative phase using a second-order finite difference stencil.
    cum_phase_derivative = compute_first_derivative_in_time(time, cum_phase)

    return time, cum_phase, amplitude, cum_phase_derivative


def fit_quadratic_to_omega_and_find_minimum(
    r_over_M: float, time: NDArray[np.float64], omega: NDArray[np.float64]
) -> float:
    """
    Fits a quadratic curve to the filtered omega data within a specified time range and outputs the minimum omega value.

    :param time: A numpy array containing time values.
    :param omega: A numpy array containing omega values corresponding to the time values.

    :return: The absolute value of the quadratic curve evaluated at t=0.

    :raises ValueError: If the lengths of time and omega arrays are not equal.
    """
    if len(time) != len(omega):
        raise ValueError("The lengths of time and omega arrays must be equal.")

    def quadratic(x: float, a: float, b: float, c: float) -> float:
        """
        Represents a quadratic function.

        :param x: The independent variable.
        :param a: The coefficient of the x^2 term.
        :param b: The coefficient of the x term.
        :param c: The constant term.

        :return: The value of the quadratic function at x.
        """
        return a * x**2 + b * x + c

    # Filter the data for t=r_over_M to t=r_over_M+200
    fit_start = r_over_M
    fit_end = r_over_M + 200.0
    time_filtered = time[(time >= fit_start) & (time <= fit_end)]
    omega_filtered = omega[(time >= fit_start) & (time <= fit_end)]

    # Fit a quadratic curve to the Omega data using nonlinear least squares
    params, _ = curve_fit(quadratic, time_filtered, omega_filtered)

    # Find the extremum value of the quadratic curve
    a, b, c = params
    extremum_x = -b / (2 * a)
    omega_min_quad_fit = np.fabs(quadratic(extremum_x, a, b, c))
    omega_at_t_zero = np.fabs(quadratic(0.0, a, b, c))

    print(
        f"The extremum of the quadratic curve occurs at t = {extremum_x:.15f} "
        f"with omega = {omega_min_quad_fit:.15f}. Implied omega(t=0) = {omega_at_t_zero:.15f}"
    )

    return float(omega_at_t_zero)  # Explicitly cast to float


def perform_complex_fft(
    time: NDArray[np.float64], real: NDArray[np.float64], imag: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """
    Performs a complex Fast Fourier Transform (FFT) on the input time, real, and imaginary data.

    :param time: A numpy array containing time values.
    :param real: A numpy array containing the real part of the signal.
    :param imag: A numpy array containing the imaginary part of the signal.

    :return: A tuple containing two numpy arrays (frequencies, fft_data).

    :raises ValueError: If the lengths of time, real, and imag arrays are not equal.
    """

    if not len(time) == len(real) == len(imag):
        raise ValueError("The lengths of time, real, and imag arrays must be equal.")

    # Combine the real and imaginary data into a single complex signal
    complex_signal = real + 1j * imag

    # Perform the complex FFT
    fft_data = np.fft.fft(complex_signal)

    # Calculate the frequency values
    dt = time[1] - time[0]
    n = len(time)
    frequencies = np.fft.fftfreq(n, d=dt)

    return frequencies, fft_data


def main() -> None:
    """
    Main function that reads the gravitational wave data file and the dimensionless
    radius r/M, processes the data, and saves the output to a file. The input filename
    and r/M value are provided via the command line.
    """
    if len(sys.argv) != 2:
        raise RuntimeError(
            "Usage: python3 psi4_to_phase_amp_omega_FFI_strain_psi4check.py <gravitational_wave_data.asc or .txt>"
        )

    file_name = sys.argv[1]

    if not (file_name.endswith(".asc") or file_name.endswith(".txt")):
        raise ValueError("Error: Input file must have a '.asc' or '.txt' extension.")

    r_over_M, time, mode_data = read_BHaH_psi4_file(file_name)
    real, imag = mode_data[(2, 2)]
    # print(real, imag)

    time, cumulative_phase, amplitude, omega = compute_psi4_wave_phase_and_amplitude(
        time, real, imag
    )

    output_file = (
        file_name.replace(".asc", "_phase_amp_omega.asc")
        if file_name.endswith(".asc")
        else file_name.replace(".txt", "_phase_amp_omega.txt")
    )

    with open(output_file, mode="w", encoding="utf-8") as file:
        file.write("# Time    cumulative_phase    amplitude    omega\n")
        for t, cp, a, o in zip(time, cumulative_phase, amplitude, omega):
            file.write(f"{t:.15f} {cp:.15f} {a:.15f} {o:.15f}\n")

    print(f"Processed data has been saved to {output_file}")

    min_omega = fit_quadratic_to_omega_and_find_minimum(r_over_M, time, omega)

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
    if file_name.endswith(".asc"):
        output_file = output_file.replace(".asc", "_strain.asc")
    elif file_name.endswith(".txt"):
        output_file = output_file.replace(".txt", "_strain.txt")

    with open(output_file, mode="w", encoding="utf-8") as file:
        file.write("# Time    Second_Integral_Real    Second_Integral_Imag\n")
        for t, real, imag in zip(time, second_integral_real, second_integral_imag):
            file.write(f"{t:.15f} {real:.15f} {imag:.15f}\n")

    print(f"Second time integral data has been saved to {output_file}")

    # Calculate the second time derivative of second_integral_real and second_integral_imag
    second_derivative_real = compute_second_derivative_in_time(
        time, second_integral_real
    )
    second_derivative_imag = compute_second_derivative_in_time(
        time, second_integral_imag
    )
    with open("check.txt", mode="w", encoding="utf-8") as file:
        file.write("# Time    Second_Integral_Real    Second_Integral_Imag\n")
        for t, real, imag in zip(time, second_derivative_real, second_derivative_imag):
            file.write(f"{t:.15f} {real:.15f} {imag:.15f}\n")


if __name__ == "__main__":
    # First run doctests
    import doctest

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")

    # Then run the main() function.
    main()
