import argparse
import sys
import os


DEFAULT_MAG_ERROR_COLUMN = "Source_AMag_Err_T1"


def _trim_and_validate_sequence(
    sequence_df,
    *,
    sequence_length,
    monotonic_length
):
    candidate_run_df = sequence_df.copy().reset_index(drop=True)
    if candidate_run_df.empty:
        return None

    if sequence_length <= 0 or sequence_length % 2 == 0:
        raise ValueError("sequence_length must be a positive odd integer.")

    if monotonic_length < 0:
        raise ValueError("monotonic_length must be non-negative.")

    half_window = sequence_length // 2
    center_idx = candidate_run_df['Source_AMag_T1'].idxmax()
    start_idx = center_idx - half_window
    end_idx = center_idx + half_window

    if start_idx < 0 or end_idx >= len(candidate_run_df):
        return None

    candidate_df = candidate_run_df.iloc[start_idx:end_idx + 1].reset_index(drop=True)

    if monotonic_length > 0:
        max_monotonic_length = half_window + 1
        if monotonic_length > max_monotonic_length:
            raise ValueError(
                "monotonic_length must not exceed (sequence_length + 1) // 2."
            )
        mags = candidate_df['Source_AMag_T1']
        first_segment = mags.iloc[:monotonic_length]
        last_segment = mags.iloc[-monotonic_length:]
        first_monotonic = first_segment.is_monotonic_increasing
        last_monotonic = last_segment.is_monotonic_decreasing
        if not (first_monotonic and last_monotonic):
            return None

    return candidate_df


def _analyze_sequence(
    sequence_df,
    *,
    sequence_length,
    monotonic_length,
    mag_error_column,
):
    import numpy as np

    from kwee_and_van_woerden import kwee_van_woerden

    candidate_df = _trim_and_validate_sequence(
        sequence_df,
        sequence_length=sequence_length,
        monotonic_length=monotonic_length
    )

    if candidate_df is None:
        return None

    tom, tom_uncertainty = kwee_van_woerden(
        candidate_df['BJD_TDB'].values,
        candidate_df['Source_AMag_T1'].values
    )

    output_df = candidate_df[['BJD_TDB', 'Source_AMag_T1']].copy()
    output_df['TOM'] = np.nan
    output_df['TOM_uncertainty'] = np.nan
    center_idx = output_df['Source_AMag_T1'].idxmax()
    output_df.at[center_idx, 'TOM'] = tom
    output_df.at[center_idx, 'TOM_uncertainty'] = tom_uncertainty

    return {
        "candidate_df": candidate_df,
        "output_df": output_df,
        "tom": float(tom),
        "tom_uncertainty": float(tom_uncertainty),
        "integer_date": int(np.floor(tom)),
        "mag_error_column": mag_error_column,
    }


def _pdf_output_path(output_file, integer_date):
    output_dir = os.path.dirname(output_file)
    output_name = os.path.basename(output_file)
    base_name, _ = os.path.splitext(output_name)
    return os.path.join(output_dir, f"{base_name}_{integer_date}.pdf")


def _infer_plot_labels(input_file, star_name=None, filter_name=None):
    input_name = os.path.splitext(os.path.basename(input_file))[0]
    tokens = input_name.replace("-", "_").split("_")

    inferred_filter = None
    if filter_name is None:
        for token in reversed(tokens):
            token_upper = token.upper()
            if len(token_upper) == 1 and token_upper.isalpha():
                inferred_filter = token_upper
                break
            if token_upper in {"B", "V", "R", "I"}:
                inferred_filter = token_upper
                break

    inferred_star = None
    if star_name is None:
        if inferred_filter is not None and inferred_filter in tokens:
            filter_index = tokens.index(inferred_filter)
            star_tokens = tokens[:filter_index]
        elif inferred_filter is not None and inferred_filter.lower() in tokens:
            filter_index = tokens.index(inferred_filter.lower())
            star_tokens = tokens[:filter_index]
        else:
            star_tokens = tokens

        cleaned_tokens = []
        for token in star_tokens:
            if token.lower() in {"measurement", "table", "cleaned", "minima", "lc"}:
                break
            if token:
                cleaned_tokens.append(token)
        if cleaned_tokens:
            inferred_star = " ".join(cleaned_tokens)

    final_star = star_name if star_name is not None else inferred_star
    final_filter = filter_name if filter_name is not None else inferred_filter

    if final_star is None:
        final_star = "Target"
    if final_filter is None:
        final_filter = "Magnitude"

    return final_star, final_filter


def _x_axis_label(integer_date, tom, tom_uncertainty):
    return (
        f"BJD_TDB - {integer_date} "
        f"(T0 = {tom:.6f}, SE = {tom_uncertainty:.6f})"
    )


def _y_axis_label(star_name, filter_name):
    if filter_name == "Magnitude":
        return f"{star_name} Magnitude"
    return f"{star_name} {filter_name} Magnitude"


def _render_minimum_visualization(minimum_result, pdf_output_file):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    candidate_df = minimum_result["candidate_df"]
    tom = minimum_result["tom"]
    tom_uncertainty = minimum_result["tom_uncertainty"]
    integer_date = minimum_result["integer_date"]
    mag_error_column = minimum_result["mag_error_column"]
    star_name = minimum_result["star_name"]
    filter_name = minimum_result["filter_name"]

    times = candidate_df["BJD_TDB"].to_numpy(dtype=float)
    mags = candidate_df["Source_AMag_T1"].to_numpy(dtype=float)
    mag_errs = candidate_df[mag_error_column].to_numpy(dtype=float)

    x_obs = times - integer_date
    x_mirror = (2.0 * tom - times) - integer_date
    x_tom = tom - integer_date

    obs_poly = np.poly1d(np.polyfit(x_obs, mags, 2))
    mirror_poly = np.poly1d(np.polyfit(x_mirror, mags, 2))

    x_min = min(x_obs.min(), x_mirror.min())
    x_max = max(x_obs.max(), x_mirror.max())
    x_fit = np.linspace(x_min, x_max, 400)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.errorbar(
        x_obs,
        mags,
        yerr=mag_errs,
        fmt="o",
        color="#1f5a84",
        ecolor="#7aa6c2",
        elinewidth=1.0,
        capsize=3,
        markersize=5,
        label="Observations",
    )
    ax.errorbar(
        x_mirror,
        mags,
        yerr=mag_errs,
        fmt="o",
        color="#2a8a2a",
        ecolor="#8ec58e",
        elinewidth=1.0,
        capsize=3,
        markersize=5,
        alpha=0.9,
        label="Folded Observations",
    )
    ax.plot(
        x_fit,
        obs_poly(x_fit),
        linestyle=":",
        linewidth=1.8,
        color="#1f5a84",
        label="Poly. (Observations)",
    )
    ax.plot(
        x_fit,
        mirror_poly(x_fit),
        linestyle=":",
        linewidth=1.8,
        color="#2a8a2a",
        label="Poly. (Folded Observations)",
    )
    ax.axvline(
        x_tom,
        color="#1f5a84",
        linestyle=(0, (4, 4)),
        linewidth=1.4,
        label="Minima",
    )
    ax.set_title("Kwee and van Woerden Visualization")
    ax.set_xlabel(_x_axis_label(integer_date, tom, tom_uncertainty))
    ax.set_ylabel(_y_axis_label(star_name, filter_name))
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(pdf_output_file, format="pdf")
    plt.close(fig)


def _write_minimum_result(
    minimum_result,
    *,
    is_first_valid_minimum,
    csvfile,
    output_file,
):
    if not is_first_valid_minimum:
        csvfile.write('\n')

    minimum_result["output_df"].to_csv(
        csvfile,
        index=False,
        header=is_first_valid_minimum,
    )

    pdf_output_file = _pdf_output_path(
        output_file,
        minimum_result["integer_date"],
    )
    _render_minimum_visualization(minimum_result, pdf_output_file)

    return True

def find_variable_star_minima(
    input_file,
    output_file,
    sequence_length=7,
    monotonic_length=None,
    night_mode='land',
    night_gap_hours=8.0,
    mag_error_column=DEFAULT_MAG_ERROR_COLUMN,
    star_name=None,
    filter_name=None,
):
    """
    Reads an AstroImageJ Measurements file and finds clean, variable star minima.

    A 'clean' minimum must satisfy three conditions:
    1. It belongs to a contiguous sequence of observations with magnitudes above the
       dataset median within a nightly segment.
    2. It contains exactly `sequence_length` samples, where `sequence_length` is odd
       and the middle sample is the maximum magnitude in the candidate sequence.
    3. The first `monotonic_length` magnitudes are non-decreasing and the last
       `monotonic_length` are non-increasing.

    Args:
        input_file (str): Path to the AstroImageJ Measurements file.
        output_file (str): Path to the output CSV file.
        sequence_length (int): The total number of data points for a valid minimum. Must be a positive odd integer.
        monotonic_length (int | None): The number of points at the start and end of the
                                       sequence to check for monotonic behavior. Defaults
                                       to (sequence_length + 1) // 2 when not provided.
        night_mode (str): 'land' splits nights by gaps exceeding `night_gap_hours`; 'space'
                          treats the light curve as a single continuous segment.
        night_gap_hours (float): Time gap (in hours) that defines separate nights in land mode.
        mag_error_column (str): Column containing magnitude uncertainties for the PDF
                                visualizations.
        star_name (str | None): Override for the star name used in plot labels.
        filter_name (str | None): Override for the filter/band used in plot labels.
    """
    import pandas as pd

    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    required_columns = ['BJD_TDB', 'Source_AMag_T1', mag_error_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Input file is missing required columns: {', '.join(missing_columns)}"
        )
    if df.empty:
        raise ValueError("Input file contains no observations.")
    if df[required_columns].isna().any().any():
        raise ValueError("Input file contains missing values in required columns.")

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='raise')

    df.sort_values(by='BJD_TDB', inplace=True)
    df.reset_index(drop=True, inplace=True)

    median_mag = df['Source_AMag_T1'].median()

    night_mode_normalized = (night_mode or '').lower()
    if night_mode_normalized not in {'land', 'space'}:
        raise ValueError("night_mode must be 'land' or 'space'.")
    if sequence_length <= 0 or sequence_length % 2 == 0:
        raise ValueError("sequence_length must be a positive odd integer.")
    if monotonic_length is None:
        monotonic_length = (sequence_length + 1) // 2
    if monotonic_length < 0:
        raise ValueError("monotonic_length must be non-negative.")
    if monotonic_length > (sequence_length + 1) // 2:
        raise ValueError(
            "monotonic_length must not exceed (sequence_length + 1) // 2."
        )

    if night_mode_normalized == 'land':
        if night_gap_hours <= 0:
            raise ValueError("night_gap_hours must be positive for land mode.")
        gap_days = night_gap_hours / 24.0
        diffs = df['BJD_TDB'].diff().fillna(0)
        df['segment'] = (diffs > gap_days).cumsum()
    else:
        df['segment'] = 0

    inferred_star_name, inferred_filter_name = _infer_plot_labels(
        input_file,
        star_name=star_name,
        filter_name=filter_name,
    )

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', newline='') as csvfile:
        is_first_valid_minimum = True

        for _, segment_df in df.groupby('segment'):
            segment_df = segment_df.reset_index(drop=True)

            above_median = segment_df['Source_AMag_T1'] > median_mag
            start_idx = None

            for i, flag in enumerate(above_median):
                if flag and start_idx is None:
                    start_idx = i
                elif not flag and start_idx is not None:
                    end_idx = i - 1
                    minimum_result = _analyze_sequence(
                        segment_df.iloc[start_idx:end_idx + 1],
                        sequence_length=sequence_length,
                        monotonic_length=monotonic_length,
                        mag_error_column=mag_error_column,
                    )
                    if minimum_result is not None:
                        minimum_result["star_name"] = inferred_star_name
                        minimum_result["filter_name"] = inferred_filter_name
                        wrote = _write_minimum_result(
                            minimum_result,
                            is_first_valid_minimum=is_first_valid_minimum,
                            csvfile=csvfile,
                            output_file=output_file,
                        )
                        is_first_valid_minimum = False
                    start_idx = None

            if start_idx is not None:
                minimum_result = _analyze_sequence(
                    segment_df.iloc[start_idx:],
                    sequence_length=sequence_length,
                    monotonic_length=monotonic_length,
                    mag_error_column=mag_error_column,
                )
                if minimum_result is not None:
                    minimum_result["star_name"] = inferred_star_name
                    minimum_result["filter_name"] = inferred_filter_name
                    wrote = _write_minimum_result(
                        minimum_result,
                        is_first_valid_minimum=is_first_valid_minimum,
                        csvfile=csvfile,
                        output_file=output_file,
                    )
                    is_first_valid_minimum = False

    if is_first_valid_minimum:
        print("Processing complete. No clean minima were found that met all criteria.")
    else:
        print(f"Successfully processed the file and saved the results to {output_file}")


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Find clean variable-star minima from an AstroImageJ measurements CSV "
            "using a global-median threshold and a symmetric odd-length window."
        ),
        usage=(
            "%(prog)s <input_file.csv> {land,space} <sequence_length> "
            "[--output OUTPUT_FILE] [--monotonic-length N] [--night-gap-hours HOURS] "
            "[--mag-error-column COLUMN] [--star-name STAR] [--filter-name FILTER]"
        )
    )
    parser.add_argument(
        "input_file",
        help="Path to the AstroImageJ measurements CSV file. Bare filenames are resolved from input/."
    )
    parser.add_argument(
        "night_mode",
        choices=["land", "space"],
        help="Use 'land' to split nights by time gaps or 'space' to treat the file as one segment."
    )
    parser.add_argument(
        "sequence_length",
        type=int,
        help="Odd number of observations to keep, centered on the dimmest selected observation."
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        help="Path to the output CSV file. Defaults to output/<input_stem>_minima.csv."
    )
    parser.add_argument(
        "--monotonic-length",
        type=int,
        default=None,
        help="Override the monotonic edge length. Defaults to (sequence_length + 1) // 2."
    )
    parser.add_argument(
        "--night-gap-hours",
        type=float,
        default=8.0,
        help="Gap in hours used to split nights in land mode. Default: 8.0."
    )
    parser.add_argument(
        "--mag-error-column",
        default=DEFAULT_MAG_ERROR_COLUMN,
        help=(
            "Column containing magnitude uncertainties for PDF error bars. "
            f"Default: {DEFAULT_MAG_ERROR_COLUMN}."
        ),
    )
    parser.add_argument(
        "--star-name",
        default=None,
        help="Override the star name used in plot labels. Default: inferred from input filename.",
    )
    parser.add_argument(
        "--filter-name",
        default=None,
        help="Override the filter/band name used in plot labels. Default: inferred from input filename.",
    )
    return parser


def _resolve_input_path(input_file):
    if os.path.dirname(input_file):
        return input_file
    return os.path.join("input", input_file)


def _default_output_path(input_file):
    input_name = os.path.basename(input_file)
    base_name, _ = os.path.splitext(input_name)
    return os.path.join("output", f"{base_name}_minima.csv")

if __name__ == "__main__":
    parser = _build_argument_parser()
    if len(sys.argv) == 1:
        parser.print_usage()
        sys.exit(1)

    args = parser.parse_args()

    input_file = _resolve_input_path(args.input_file)

    output_file = args.output_file
    if output_file is None:
        output_file = _default_output_path(args.input_file)

    find_variable_star_minima(
        input_file,
        output_file,
        sequence_length=args.sequence_length,
        monotonic_length=args.monotonic_length,
        night_mode=args.night_mode,
        night_gap_hours=args.night_gap_hours,
        mag_error_column=args.mag_error_column,
        star_name=args.star_name,
        filter_name=args.filter_name,
    )
