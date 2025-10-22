import sys
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

# Read input
data = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        break
    parts = line.split()
    if len(parts) != 2:
        sys.stderr.write("Error: Invalid input format.\n")
        sys.exit(1)
    try:
        t = float(parts[0])
        m = float(parts[1])
        data.append((t, m))
    except ValueError:
        sys.stderr.write("Error: Invalid numeric values.\n")
        sys.exit(1)

if not data:
    sys.stderr.write("Error: No data provided.\n")
    sys.exit(1)

# Sort by time
data.sort()
times, mags = zip(*data)
times = np.array(times)
mags = np.array(mags)
N = len(times)

if N % 2 == 0:
    sys.stderr.write("Error: Please provide an odd number of data points.\n")
    sys.exit(1)

if N < 3:
    sys.stderr.write("Error: At least 3 data points required.\n")
    sys.exit(1)

n = (N - 1) // 2

# Separate integer and fractional parts for stability
base = np.floor(np.min(times))
frac_times = times - base

t_min = frac_times[0]
t_max = frac_times[-1]

if t_min == t_max:
    sys.stderr.write("Error: All times are identical.\n")
    sys.exit(1)

h = (t_max - t_min) / (2 * n)
equal_times = np.array([t_min + i * h for i in range(2 * n + 1)])
equal_mags = np.interp(equal_times, frac_times, mags)

# Fit quadratic to fractional times to find initial To
coef = np.polyfit(frac_times, mags, 2)
a, b, c = coef
if a >= 0:
    sys.stderr.write("Error: Quadratic fit does not indicate a maximum (a >= 0).\n")
    sys.exit(1)
initial_To = -b / (2 * a)

# Set delta for trial epochs
d = h / 2.0

# Loop to ensure middle s is the smallest
max_iterations = 100
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
    sys.stderr.write("Error: Failed to converge to a local minimum for s.\n")
    sys.exit(1)

# Now fit parabola to the three s values
T_vals = np.array([initial_To - d, initial_To, initial_To + d])
s_vals = np.array([s_left, s_mid, s_right])
coef_s = np.polyfit(T_vals, s_vals, 2)
A, B, C = coef_s
if A <= 0:
    sys.stderr.write("Error: Parabola for s does not open upward (A <= 0).\n")
    sys.exit(1)

epoch_frac = -B / (2 * A)
epoch = base + epoch_frac

# Standard error
Z = n
if Z - 1 <= 0:
    sys.stderr.write("Error: Too few pairs for standard error calculation.\n")
    sys.exit(1)
numer = 4 * A * C - B ** 2
denom = 4 * A ** 2 * (Z - 1)
if denom <= 0 or numer < 0:
    sys.stderr.write("Error: Invalid values for standard error calculation.\n")
    sys.exit(1)
std_err = (numer / denom) ** 0.5

# Output
print(f"{epoch} {std_err}")