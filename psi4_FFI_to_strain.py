"""
This module processes gravitational wave data from numerical relativity simulations.

The module reads a directory of ASCII files of various ell values of waveform data, computes phase and amplitude data,
calculates a minimum data frequency using a quadratic fit of the monotonic phase of  ell2 m2 data,
and uses a Fast Fourier Transform to compute the second time integral of the waveform, (the strain).
The module computes a second derivative of the result to check against the original data.

The phase and amplitude, the second integral, and the twice integrated-twice differentiated data is saved to txt files.
The primary function returns the second integral data as a numpy array with the various ell-values.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

import sys
import os
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


INPUT_DIR = "../BH_VIS/data/r100" # changeable in animation_main.py
OUTPUT_DIR = "../BH_VIS/data/r100_strain"
FILE_PATTERN = "_l[MODE=L]-"
ELL_MIN = 2
ELL_MAX = 8
EXT_RAD = 100 # changeable in animation_main.py
INTERVAL = 200
CUTTOFF_FACTOR = 0.75
STATUS_MESSAGES = True
WRITE_FILES = True # changeable in animation_main.py


def read_psi4_dir() -> tuple[np.ndarray, np.ndarray]:
    """
    Reads data from psi4 output directory and returns time and mode data.

    :return: tuple[np.ndarray, np.ndarray]
        - time_data: Array of numpy.float64 time values (shape: (n_times,) ).
        - mode_data: 2D Array for modes of numpy.complex128 data (shape: (2*l+1, n_times,) ).
    """

    time_data: list[np.ndarray] = []
    psi4_modes_data: list[np.ndarray] = []
    n_times = -1

    for ell in range(ELL_MIN, ELL_MAX + 1):
        filepath = find_file_for_l(ell)
        with open(filepath, "r", encoding="utf-8") as file:
            lines = [line for line in file.readlines() if not line.startswith("#")]
        data = np.array([np.array(line.split(), dtype=np.float64) for line in lines])

        time_data, index = np.unique(
            data[:, 0], return_index=True
        )  # sorts by time, removing duplicates
        data = data[index]  # sorts data accordningly

        if n_times == -1:
            n_times = len(time_data)

        if n_times != len(time_data):
            raise ValueError(
                f"Inconsistent times for l={ell}. Expected {n_times}, got {len(time_data)}."
            )

        real_idx = 1
        for _ in range(2 * ell + 1):
            psi4_modes_data.append(data[:, real_idx] + 1j * data[:, real_idx + 1])
            real_idx += 2
    return np.array(time_data), np.array(psi4_modes_data)


def psi4_ffi_to_strain():
    """
    Calculates the strain modes from PSI4 data using FFI.

    Returns:
        A numpy array of numpy arrays representing the strain modes.

    Raises:
        IOError: If there is an error reading the PSI4 data or writing strain data.
        ValueError: If the lengths of the time and data arrays are not equal.
    """

    try:
        time_arr, psi4_modes_data = read_psi4_dir()
    except IOError as e:
        raise IOError(f"Error reading PSI4 data: {e}") from e

    # Get minimum frequency cutoff from l=m=2 mode
    min_omega_l2m2 = extract_min_omega_ell2_em2(
        time_arr, psi4_modes_data[get_index_from_modes(2, 2)]
    )
    freq_cutoff = CUTTOFF_FACTOR * min_omega_l2m2

    # Initialize arrays for strain modes and their second time derivatives
    strain_modes = np.zeros_like(psi4_modes_data)
    strain_modes_ddot = np.zeros_like(psi4_modes_data)

    # Calculate frequency list for FFT
    freq_list = np.fft.fftfreq(len(time_arr), time_arr[1] - time_arr[0]) * 2 * np.pi

    # Next loop over modes and perform an FFT:
    mode_idx = 0
    for ell in range(ELL_MIN, ELL_MAX + 1):
        for em in range(-ell, ell + 1):
            # Apply FFT and filter, see Eq. 27 in https://arxiv.org/abs/1006.1632
            fft_result = np.fft.fft(psi4_modes_data[get_index_from_modes(ell, em)])
            for i, freq in enumerate(freq_list):
                if np.fabs(freq) <= np.fabs(freq_cutoff):
                    fft_result[i] *= 1 / (1j * freq_cutoff) ** 2
                else:
                    fft_result[i] *= 1 / (1j * freq) ** 2
            # Inverse FFT to get strain
            strain_modes[mode_idx] = np.fft.ifft(fft_result)

            # Calculate second time derivative
            strain_modes_ddot[mode_idx] = second_time_derivative(
                time_arr, strain_modes[mode_idx]
            )
            mode_idx += 1

    # Save the strain output to a file with _conv_to_strain.txt extension
    if OUTPUT_DIR != "":
        for ell in range(ELL_MIN, ELL_MAX + 1):
            strain_filename = f"Rpsi4_r{EXT_RAD:06.1f}_l{ell}_conv_to_strain.txt"
            ddot_filename = f"Rpsi4_r{EXT_RAD:06.1f}_l{ell}_from_strain.txt"
            labels = []
            strain_cols = []
            ddot_cols = []
            col = 0

            labels.append(f"# column {col}: t-R_ext = [retarded time]")
            strain_cols.append(time_arr)
            col += 1

            for em in range(-ell, ell + 1):
                mode_data = strain_modes[get_index_from_modes(ell, em)]
                ddot_data = strain_modes_ddot[get_index_from_modes(ell, em)]

                labels.append(f"# column {col}: Re(h_{{l={ell},m={em}}}) * R_ext")
                strain_cols.append(mode_data.real)
                ddot_cols.append(ddot_data.real)
                col += 1

                labels.append(f"# column {col}: Im(h_{{l={ell},m={em}}}) * R_ext")
                strain_cols.append(mode_data.imag)
                ddot_cols.append(ddot_data.imag)
                col += 1

            arrays_to_txt(labels, strain_cols, strain_filename, OUTPUT_DIR)
            arrays_to_txt(labels, ddot_cols, ddot_filename, OUTPUT_DIR)

    return time_arr, strain_modes


def find_file_for_l(ell: int) -> str:
    """
    Finds the file path with the corresponding ell value in the given directory.

    :param ell: (int): l mode to search for.
    :return: Path to the found file.
    :raises FileNotFoundError: If no file matching the pattern is found.
    """

    for filename in os.listdir(INPUT_DIR):
        if FILE_PATTERN.replace("[MODE=L]", f"{ell}") in filename:
            return os.path.join(INPUT_DIR, filename)
    raise FileNotFoundError(f"File with mode l={ell} not found.")


def get_index_from_modes(ell: int, em: int, ell_min=ELL_MIN) -> int:
    """
    Returns the array index for mode data given (ell, em).
    The index begins with 0 and through m (inner loop) then l (outer loop).

    :param ell: The l Spherical harmonics mode number
    :param em: The m Spherical harmonics mode number
    :param ell_min: The minimum ell value used in the array (default is ELL_MIN).

    :return: The mode data array index for (ell, em).

    >>> get_index_from_modes(3, 1, 2)
    9
    """
    return ell**2 + ell + em - ell_min**2


def get_modes_from_index(idx: int, ell_min=ELL_MIN) -> Tuple[int, int]:
    """
    Returns the (ell, em) mode numbers given the array index.

    :param idx: The mode data array index.
    :param ell_min: The minimum ell value used in the array (default is ELL_MIN).

    :return: A tuple containing the (ell, em) mode numbers.

    >>> get_modes_from_index(9, 2)
    (3, 1)
    """
    idx += ell_min**2
    ell = int(np.sqrt(idx))
    em = idx - ell**2 - ell
    return ell, em


def arrays_to_txt(
    labels: List[str],
    collection: List[np.ndarray],
    filename: str,
    dir_path: str,
) -> None:
    """Writes an array of NumPy arrays to a text file, formatting each row with labels.

    Args:
        labels: A list of comment lines.
        collection: A list of NumPy arrays, where each inner array represents a column.
        filename: The name of the file to write to.
        dir_path: The path to the directory where the file will be saved.

    Raises:
        IOError: If there is an error creating the directory or writing to the file.
    """
    if WRITE_FILES:
        try:
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, filename)

            with open(file_path, mode="w", encoding="utf-8") as file:
                file.write("".join([f"{label}\n" for label in labels]))
                for row in zip(*collection):
                    file.write(" ".join([f"{item:.15f}" for item in row]) + "\n")
            print(f"File {filename} saved to {dir_path}")
        except IOError as e:
            raise IOError(f"Error saving data to file: {e}") from e


def first_time_derivative(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate the time derivative of the input data using a second-order finite difference stencil.

    :param time: A numpy array containing time values.
    :param data: A numpy array containing the data to be differentiated.
    :return: A numpy array containing the time derivative of the input data.

    >>> time = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    >>> data = np.array([0, 1, 4, 9, 16], dtype=np.float64)
    >>> first_time_derivative(time, data)
    array([1., 2., 4., 6., 7.])
    """

    delta_t = time[1] - time[0]
    data_dt = np.zeros_like(data)

    # Second-order in the interior:
    data_dt[1:-1] = (data[2:] - data[:-2]) / (2 * delta_t)

    # Drop to first-order at the endpoints
    data_dt[0] = (data[1] - data[0]) / delta_t
    data_dt[-1] = (data[-1] - data[-2]) / delta_t

    return data_dt


def second_time_derivative(
    time: NDArray[np.float64], data: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the second time derivative of the input data using the second-order
    finite difference method, with upwind/downwind stencils for the endpoints.

    :param time: A numpy array containing time values.
    :param data: A numpy array containing corresponding function values to take derivatives of.
    :return: A numpy array containing the second time derivative of the function data.

    >>> time = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    >>> data = np.array([0, 1, 4, 9, 16], dtype=np.float64)
    >>> second_time_derivative(time, data)
    array([2., 2., 2., 2., 2.])
    """
    delta_t = time[1] - time[0]
    data_dtdt = np.zeros_like(data)

    # Interior points using central finite difference
    central = data[:-2] - 2 * data[1:-1] + data[2:]
    data_dtdt[1:-1] = central / delta_t**2

    # Endpoint 0: forward finite difference (downwind)
    forward = 2 * data[0] - 5 * data[1] + 4 * data[2] - data[3]
    data_dtdt[0] = forward / delta_t**2

    # Endpoint n-1: backward finite difference (upwind)
    backward = 2 * data[-1] - 5 * data[-2] + 4 * data[-3] - data[-4]
    data_dtdt[-1] = backward / delta_t**2

    return data_dtdt


def psi4_phase_and_amplitude(
    time: NDArray[np.float64], cmplx: NDArray[np.complex128]
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Calculates the amplitude and cumulative phase of a gravitational wave signal.

    :param time: A numpy array containing time values.
    :param cmplx: A numpy array containing the a complex signal.
    :return: A tuple containing four numpy arrays:
        (time, amplitude, cumulative_phase, cumulative_phase_derivative).
    :raises ValueError: If the lengths of time, real, and imag arrays are not equal.
    """
    if len(time) != len(cmplx):
        raise ValueError(
            f"Time {len(time)} and data {len(cmplx)} array lengths must be equal."
        )

    amplitudes = np.abs(cmplx)
    phases = np.angle(cmplx)
    cycles = 0
    cum_phase = np.zeros_like(time)
    last_phase = phases[0]

    for i, ph in enumerate(phases):
        # identify phase wrapping
        if np.abs(ph - last_phase) >= np.pi:
            cycles += -1 if ph > 0 else 1
        cum_phase[i] = ph + 2 * np.pi * cycles
        last_phase = ph

    cum_phase_dt = first_time_derivative(time, cum_phase)
    return time, amplitudes, cum_phase, cum_phase_dt


def quad_fit_intercept(time: NDArray[np.float64], data: NDArray[np.float64]) -> float:
    """
    Samples data from a time interval, applies a quadratic fit, and outputs the |y-intercept|.
    This function is intended for l=m=2 angular frequency data.

    :param interval_start: A float specifying the begining of the sample interval.
    :param time: A numpy array containing time values.
    :param data: A numpy array containing data values corresponding to the time values.
    :return: The float absolute value of the quadratic curve evaluated at t=0.
    :raises ValueError: If the lengths of time and data arrays are not equal.
    """
    if len(time) != len(data):
        raise ValueError(
            f"Time {len(time)} and data {len(data)} array lengths must be equal."
        )

    def quadratic(x: float, a: float, b: float, c: float) -> float:
        """
        Evaluates a quadratic (ax^2 + bx + c)

        :param x: The independent variable.
        :param a: The coefficient of the x^2 term.
        :param b: The coefficient of the x term.
        :param c: The constant term.

        :return: The value of the quadratic function at x.
        """
        return a * x**2 + b * x + c

    # Re-index, keeping only the intersection between numpy arrays
    time_filtered = time[(EXT_RAD <= time) & (time <= EXT_RAD + INTERVAL)]
    data_filtered = data[(EXT_RAD <= time) & (time <= EXT_RAD + INTERVAL)]

    # Fit a quadratic curve to the data using nonlinear least squares
    params, *_ = curve_fit(quadratic, time_filtered, data_filtered)

    # Find the extremum value of the quadratic curve
    a, b, c = params
    extremum_x = -b / (2 * a)
    quad_fit_extremum = float(quadratic(extremum_x, a, b, c))
    if STATUS_MESSAGES:
        print(
            f"Quadratic Vertex at (time = {extremum_x:.7e}, value = {quad_fit_extremum:.7e}).\n"
            f"Params: a = {a:.7e}, b = {b:.7e}, c = {c:.7e}, Intercept magnitude: {np.fabs(c):.7e}"
        )
    return np.fabs(c)


def extract_min_omega_ell2_em2(
    time: NDArray[np.float64], data_m2_l2: NDArray[np.complex128]
) -> float:
    """
    Extracts and saves the phase, amplitude, and omega data for l=m=2 mode from psi4 wave.
    Also fits a quadratic to omega and finds its minimum.

    :param time_arr: Array of time data.
    :param mode_data: Dictionary containing the mode data.
    :return: float magnitude of the minimum wave frequency of the data
    """

    collection = psi4_phase_and_amplitude(time, data_m2_l2)
    angular_frequency = collection[3]
    if OUTPUT_DIR != "":
        labels = [
            "# Col 0: Time",
            "# Col 1: Amplitude",
            "# Col 2: Cumulative_Phase",
            "# Col 3: Angular Frequency",
        ]
        filename = f"Rpsi4_r{EXT_RAD:06.1f}_ell2_m2_phase_amp_omega.txt"
        arrays_to_txt(labels, collection, filename, OUTPUT_DIR)
    return quad_fit_intercept(time, angular_frequency)


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        if STATUS_MESSAGES:
            print(f"Doctest passed: All {results.attempted} test(s) passed")

    psi4_ffi_to_strain()
