import sys
from typing import Iterable, Tuple

import numpy as np


def linear_interp(x, xp, yp):
    if x < xp[0]:
        if xp[1] == xp[0]:
            return yp[0]
        slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
        return yp[0] + slope * (x - xp[0])
    elif x > xp[-1]:
        if xp[-1] == xp[-2]:
            return yp[-1]
        slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
        return yp[-1] + slope * (x - xp[-1])
    else:
        return np.interp(x, xp, yp)


def compute_s(T, equal_times, equal_mags, h, n):
    s = 0.0
    for k in range(1, n + 1):
        tp = T + k * h
        tm = T - k * h
        mp = linear_interp(tp, equal_times, equal_mags)
        mm = linear_interp(tm, equal_times, equal_mags)
        delta = mp - mm
        s += delta ** 2
    return s


def _kwee_van_woerden_core(
    times: np.ndarray,
    mags: np.ndarray,
    *,
    max_iterations: int = 100
) -> Tuple[float, float]:
    """
    Apply the Kwee & van Woerden method to estimate the time of minimum.

    Args:
        times: Iterable of observation times.
        mags: Iterable of magnitudes corresponding to `times`.
        max_iterations: Maximum number of iterations when bracketing the minimum.

    Returns:
        Tuple containing the estimated epoch (time of minimum) and its standard error.

    Raises:
        ValueError: If inputs are invalid or the method cannot converge.
    """
    times = np.asarray(times, dtype=float)
    mags = np.asarray(mags, dtype=float)

    if times.size == 0:
        raise ValueError("No data provided.")

    if times.size != mags.size:
        raise ValueError("Times and magnitudes must have the same length.")

    order = np.argsort(times)
    times = times[order]
    mags = mags[order]

    order = np.argsort(times)
    times = times[order]
    mags = mags[order]

    N = times.size

    if N < 3:
        raise ValueError("At least 3 data points required.")

    if N % 2 == 0:
        raise ValueError("Core solver requires an odd number of data points.")

    n = (N - 1) // 2

    base = np.floor(times.min())
    frac_times = times - base

    t_min = frac_times[0]
    t_max = frac_times[-1]

    if t_min == t_max:
        raise ValueError("All times are identical.")

    h = (t_max - t_min) / (2 * n)
    if h == 0:
        raise ValueError("Sampling interval h computed as zero.")

    equal_times = np.array([t_min + i * h for i in range(2 * n + 1)])
    equal_mags = np.interp(equal_times, frac_times, mags)

    coef = np.polyfit(frac_times, mags, 2)
    a, b, _ = coef
    if a >= 0:
        raise ValueError("Quadratic fit does not indicate a maximum (a >= 0).")
    initial_To = -b / (2 * a)

    d = h / 2.0

    iteration = 0
    while iteration < max_iterations:
        s_left = compute_s(initial_To - d, equal_times, equal_mags, h, n)
        s_mid = compute_s(initial_To, equal_times, equal_mags, h, n)
        s_right = compute_s(initial_To + d, equal_times, equal_mags, h, n)

        if s_mid <= s_left and s_mid <= s_right:
            break

        if s_left < s_mid:
            initial_To -= d
        else:
            initial_To += d

        iteration += 1

    if iteration == max_iterations:
        raise ValueError("Failed to converge to a local minimum for s.")

    T_vals = np.array([initial_To - d, initial_To, initial_To + d])
    s_vals = np.array([s_left, s_mid, s_right])
    A, B, C = np.polyfit(T_vals, s_vals, 2)
    if A <= 0:
        raise ValueError("Parabola for s does not open upward (A <= 0).")

    epoch_frac = -B / (2 * A)
    epoch = base + epoch_frac

    Z = n
    if Z - 1 <= 0:
        raise ValueError("Too few pairs for standard error calculation.")
    numer = 4 * A * C - B ** 2
    denom = 4 * A ** 2 * (Z - 1)
    if denom <= 0 or numer < 0:
        raise ValueError("Invalid values for standard error calculation.")
    std_err = float(np.sqrt(numer / denom))

    return float(epoch), std_err


def kwee_van_woerden(
    times: Iterable[float],
    mags: Iterable[float],
    *,
    max_iterations: int = 100
) -> Tuple[float, float]:
    times = np.asarray(list(times), dtype=float)
    mags = np.asarray(list(mags), dtype=float)

    if times.size == 0:
        raise ValueError("No data provided.")

    if times.size != mags.size:
        raise ValueError("Times and magnitudes must have the same length.")

    order = np.argsort(times)
    times = times[order]
    mags = mags[order]

    N = times.size
    if N < 3:
        raise ValueError("At least 3 data points required.")

    if N % 2 == 1:
        return _kwee_van_woerden_core(times, mags, max_iterations=max_iterations)

    t_min = times[0]
    t_max = times[-1]
    if t_min == t_max:
        raise ValueError("All times are identical; cannot interpolate.")

    # Build evenly spaced grids within the observed range.
    grid_minus = np.linspace(t_min, t_max, N - 1)
    grid_plus = np.linspace(t_min, t_max, N + 1)

    mags_minus = np.interp(grid_minus, times, mags)
    mags_plus = np.interp(grid_plus, times, mags)

    results = []

    try:
        results.append(
            _kwee_van_woerden_core(
                grid_minus,
                mags_minus,
                max_iterations=max_iterations
            )
        )
    except ValueError:
        pass

    try:
        results.append(
            _kwee_van_woerden_core(
                grid_plus,
                mags_plus,
                max_iterations=max_iterations
            )
        )
    except ValueError:
        pass

    if not results:
        raise ValueError("Failed to evaluate even-length sequence via interpolation.")

    if len(results) == 1:
        return results[0]

    epoch_avg = sum(result[0] for result in results) / len(results)
    variance = sum(result[1] ** 2 for result in results) / len(results)
    std_combined = float(np.sqrt(variance))

    return float(epoch_avg), std_combined


def _read_stdin_pairs():
    data = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("Invalid input format.")
        try:
            t = float(parts[0])
            m = float(parts[1])
        except ValueError as exc:
            raise ValueError("Invalid numeric values.") from exc
        data.append((t, m))
    if not data:
        raise ValueError("No data provided.")
    times, mags = zip(*data)
    return times, mags


if __name__ == "__main__":
    try:
        times, mags = _read_stdin_pairs()
        epoch, std_err = kwee_van_woerden(times, mags)
        print(f"{epoch} {std_err}")
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)
