"""
This module processes gravitational wave data from numerical relativity simulations.

The module reads a directory of ASCII files of various ell values of waveform data,
computes phase and amplitude data, calculates a minimum data frequency using a
quadratic fit of the monotonic phase of  ell2 m2 data, and uses a Fast Fourier Transform
to compute the second time integral of the waveform, (the strain).
The module computes a second derivative of the result to check against the original data.

The phase and amplitude, the second integral, and the twice integrated-twice differentiated
data is saved to txt files.

The primary function returns the second integral data as a numpy array with the various ell-values.

Author: Zachariah B. Etienne
        zachetie **at** gmail **dot* com
"""

import sys
import os
from typing import Union, List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

# if data file naming conventions change then
# adjust the find_file_for_l pattern for inputs
# and the psi4_ffi_to_strain for outputs


def read_psi4_dir(
    data_dir: str, ell_max: int, ell_min: int = 2
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """
    Read data from psi4 output directory and return time and mode data.

    :return: tuple[np.ndarray, np.ndarray]
        - time_data: Array of numpy.float64 time values (shape: (n_times,) ).
        - mode_data: 2D Array for modes of numpy.complex128 data (shape: (2*l+1, n_times,) ).
    """
    time_data: NDArray[np.float64]
    psi4_modes_data: list[NDArray[np.complex128]] = []

    n_times = -1
    for ell in range(ell_min, ell_max + 1):
        filepath = find_file_for_l(data_dir, ell)
        with open(filepath, "r", encoding="utf-8") as file:
            lines = [line for line in file.readlines() if not line.startswith("#")]
        data = np.array([np.array(line.split(), dtype=np.float64) for line in lines])

        # np.unique sorts by time, removing duplicates
        time_data, indicies = np.unique(data[:, 0], return_index=True)
        data = data[indicies]  # sort data accordningly

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


def psi4_ffi_to_strain(
    data_dir: str,  # changeable in animation_main.py
    output_dir: str,
    ell_max: int = 8,
    ext_rad: float = 100.0,  # changeable in animation_main.py
    interval: float = 200.0,
    cutoff_factor: float = 0.75,
    # 0 < cutoff_factor <= 1. Lower values risk nonphysical noise, higher may filter physical data.
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """
    Calculate the strain modes from PSI4 data using FFI.

    Reads each of the ell-modes stored at `data_dir` from 2 to `ell_max` inclusive.
    Uses the ell2em2 mode to extrapolate a minimum frequency, and scales by a cutoff.
    Uses FFT and then divides by a set of frequencies to integrate the psi4 data.
    Stores the resulting strain data and its double derivative, and returns the strain data
    as an array of the various modes, each mode an array of complex data at various timestates.

    :param data_dir: directory path where raw psi4 ell-mode data is read
    :param output_dir: directory path to write files to (no output files if empty string)
    :param ell_max: maximum ell value to read data for
    :param ext_rad: extraction radius of psi4 data, used as location to sample for minimum frequency
    :param interval: the size of the sampling interval to extrapolate a minimum frequency
    :param cutoff_factor: scaling factor on the min frequency, providing a cutoff for integration
    :return: A numpy array of numpy arrays representing the strain modes.
    :raises IOError: Rrror reading the PSI4 data or writing strain data.
    :raises ValueError: Lengths of the time and data arrays are not equal.
    """
    ell_min = 2  # if not 2, also adjust calls to read_psi4_dir(), modesindex(), and indexmodes()

    try:
        time_arr, psi4_modes_data = read_psi4_dir(data_dir, ell_max)
    except IOError as e:
        raise IOError(f"Error reading PSI4 data: {e}") from e

    ell2em2_wave = psi4_phase_and_amplitude(
        time_arr, psi4_modes_data[modes_index(2, 2)]
    )

    # The estimated minimum wave frequency for ell2_em2, scaled by a factor.
    min_freq = quad_fit_intercept(time_arr, ell2em2_wave[3], ext_rad, interval)
    freq_cutoff = min_freq * cutoff_factor

    # Initialize arrays for strain modes and their second time derivatives
    strain_modes = np.zeros_like(psi4_modes_data)
    strain_modes_ddot = np.zeros_like(psi4_modes_data)

    # Calculate frequency list for FFT
    freq_list = np.fft.fftfreq(len(time_arr), time_arr[1] - time_arr[0]) * 2 * np.pi

    # Next loop over modes and perform an FFT:
    mode_idx = 0
    for ell in range(ell_min, ell_max + 1):
        for em in range(-ell, ell + 1):
            # Apply FFT and filter, see Eq. 27 in https://arxiv.org/abs/1006.1632
            fft_result = np.fft.fft(psi4_modes_data[modes_index(ell, em)])
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

    # Save ell2_em2 psi4 wave data, and all modes strain data.
    if output_dir != "":
        labels = [
            "# Col 0: Time",
            "# Col 1: Amplitude",
            "# Col 2: Cumulative_Phase",
            "# Col 3: Angular Frequency",
        ]
        filename = f"Rpsi4_r{ext_rad:06.1f}_ell2_m2_phase_amp_omega.txt"
        arrays_to_txt(labels, ell2em2_wave, filename, output_dir)

        for ell in range(ell_min, ell_max + 1):
            strain_filename = f"Rpsi4_r{ext_rad:06.1f}_l{ell}_conv_to_strain.txt"
            ddot_filename = f"Rpsi4_r{ext_rad:06.1f}_l{ell}_from_strain.txt"
            labels = []
            strain_cols = []
            ddot_cols = []
            col = 0

            labels.append(f"# column {col}: t-R_ext = [retarded time]")
            strain_cols.append(time_arr)
            col += 1

            for em in range(-ell, ell + 1):
                mode_data = strain_modes[modes_index(ell, em)]
                ddot_data = strain_modes_ddot[modes_index(ell, em)]

                labels.append(f"# column {col}: Re(h_{{l={ell},m={em}}}) * R_ext")
                strain_cols.append(mode_data.real)
                ddot_cols.append(ddot_data.real)
                col += 1

                labels.append(f"# column {col}: Im(h_{{l={ell},m={em}}}) * R_ext")
                strain_cols.append(mode_data.imag)
                ddot_cols.append(ddot_data.imag)
                col += 1

            arrays_to_txt(labels, strain_cols, strain_filename, output_dir)
            arrays_to_txt(labels, ddot_cols, ddot_filename, output_dir)

    return time_arr, strain_modes


def find_file_for_l(data_dir: str, ell: int) -> str:
    """
    Find the file path with the corresponding ell value in the given directory.

    :param ell: (int): l mode to search for.
    :return: Path to the found file.
    :raises FileNotFoundError: If no file matching the pattern is found.
    """
    for filename in os.listdir(data_dir):
        if "_l[L]-".replace("[L]", f"{ell}") in filename:
            return os.path.join(data_dir, filename)
    raise FileNotFoundError(f"File with mode l={ell} not found.")


def modes_index(ell: int, em: int, ell_min: int = 2) -> int:
    """
    Return the array index for mode data given (ell, em).

    The index begins with 0 and through m (inner loop) then l (outer loop).

    :param ell: The l Spherical harmonics mode number
    :param em: The m Spherical harmonics mode number
    :param ell_min: The minimum ell value used in the array
    :return: The mode data array index for (ell, em).

    >>> modes_index(3, 1, 2)
    9
    """
    return ell**2 + ell + em - ell_min**2


def index_modes(idx: int, ell_min: int = 2) -> Tuple[int, int]:
    """
    Given the array index, return the (ell, em) mode numbers.

    :param idx: The mode data array index.
    :param ell_min: The minimum ell value used in the array
    :return: A tuple containing the (ell, em) mode numbers.

    >>> index_modes(9, 2)
    (3, 1)
    """
    idx += ell_min**2
    ell = int(np.sqrt(idx))
    em = idx - ell**2 - ell
    return ell, em


def arrays_to_txt(
    labels: List[str],
    collection: Union[
        NDArray[np.float64], List[NDArray[np.float64]], Tuple[NDArray[np.float64], ...]
    ],
    filename: str,
    dir_path: str,
) -> None:
    """
    Write an array of NumPy arrays to a text file, formatting each row with labels.

    :param labels: A list of comment lines. Each element in the list represents a comment line.
    :param collection: An group of NumPy arrays, where each inner array represents a column.
    :param filename: The name of the file to write to.
    :param dir_path: The path to the directory where the file will be saved.
    :raises IOError: If there is an error creating the directory or writing to the file.
    """
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
    Compute the second time derivative of the input data.
    
    Uses the second-order finite difference method, with upwind/downwind stencils for the endpoints.

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
    Calculate the amplitude and cumulative phase of a gravitational wave signal.

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


def quad_fit_intercept(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    ext_rad: float,
    interval: float,
    verbose: bool = False,
) -> float:
    """
    Sample data from a time interval, apply a quadratic fit, and output the |y-intercept|.
    
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
        Evaluate a quadratic polynomial (ax^2 + bx + c).

        :param x: The independent variable.
        :param a: The coefficient of the x^2 term.
        :param b: The coefficient of the x term.
        :param c: The constant term.

        :return: The value of the quadratic function at x.
        """
        return a * x**2 + b * x + c

    # Re-index, keeping only the intersection between numpy arrays
    time_filtered = time[(ext_rad <= time) & (time <= ext_rad + interval)]
    data_filtered = data[(ext_rad <= time) & (time <= ext_rad + interval)]

    # Fit a quadratic curve to the data using nonlinear least squares
    params, *_ = curve_fit(quadratic, time_filtered, data_filtered)

    # Find the extremum value of the quadratic curve
    a, b, c = params
    extremum_x = -b / (2 * a)
    quad_fit_extremum = quadratic(extremum_x, a, b, c)
    if verbose:
        print(
            f"Quadratic Vertex at (time = {extremum_x:.7e}, value = {quad_fit_extremum:.7e}).\n"
            f"Params: a = {a:.7e}, b = {b:.7e}, c = {c:.7e}, Intercept magnitude: {np.fabs(c):.7e}"
        )
    return float(np.fabs(c))


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)

    if len(sys.argv) > 7:
        print("Error: Too many Arguments")
        MSG_1 = "Usage: psi4_ffi_to_strain.py <input dir> <output dir> "
        MSG_2 = "[maxmimum l] [extraction radius] [sample interval] [cutoff factor]"
        print(MSG_1 + MSG_2)
        sys.exit(1)

    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(workspace, "data", "GW150914_data", "r100")
    args = ["0", default_input, "", "8", "100.0", "200.0", "0.75"]
    args[: len(sys.argv)] = sys.argv

    psi4_ffi_to_strain(
        str(args[1]),
        str(args[2]),
        int(args[3]),
        float(args[4]),
        float(args[5]),
        float(args[6]),
    )
