import argparse
import sys
import os


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


def _process_sequence(
    sequence_df,
    *,
    sequence_length,
    monotonic_length,
    is_first_valid_minimum,
    csvfile
):
    import numpy as np

    from kwee_and_van_woerden import kwee_van_woerden

    candidate_df = _trim_and_validate_sequence(
        sequence_df,
        sequence_length=sequence_length,
        monotonic_length=monotonic_length
    )

    if candidate_df is None:
        return False

    try:
        tom, tom_uncertainty = kwee_van_woerden(
            candidate_df['BJD_TDB'].values,
            candidate_df['Source_AMag_T1'].values
        )
    except ValueError as ex:
        print(f"Warning: Kwee & van Woerden failed: {ex}")
        return False

    if not is_first_valid_minimum:
        csvfile.write('\n')

    output_df = candidate_df[['BJD_TDB', 'Source_AMag_T1']].copy()
    output_df['TOM'] = np.nan
    output_df['TOM_uncertainty'] = np.nan
    center_idx = output_df['Source_AMag_T1'].idxmax()
    output_df.at[center_idx, 'TOM'] = tom
    output_df.at[center_idx, 'TOM_uncertainty'] = tom_uncertainty
    output_df.to_csv(csvfile, index=False, header=is_first_valid_minimum)

    return True


def find_variable_star_minima(
    input_file,
    output_file,
    sequence_length=7,
    monotonic_length=None,
    night_mode='land',
    night_gap_hours=8.0
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
    """
    try:
        import pandas as pd

        # Read and prepare the data
        print(f"Reading file: {input_file}")
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip()

        # Ensure required columns are numeric and drop rows with invalid data
        for col in ['BJD_TDB', 'Source_AMag_T1']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['BJD_TDB', 'Source_AMag_T1'], inplace=True)


        # Sort by time to ensure chronological order before processing
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

        # Open the output file to write validated minima
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
                        wrote = _process_sequence(
                            segment_df.iloc[start_idx:end_idx + 1],
                            sequence_length=sequence_length,
                            monotonic_length=monotonic_length,
                            is_first_valid_minimum=is_first_valid_minimum,
                            csvfile=csvfile
                        )
                        if wrote:
                            is_first_valid_minimum = False
                        start_idx = None

                if start_idx is not None:
                    wrote = _process_sequence(
                        segment_df.iloc[start_idx:],
                        sequence_length=sequence_length,
                        monotonic_length=monotonic_length,
                        is_first_valid_minimum=is_first_valid_minimum,
                        csvfile=csvfile
                    )
                    if wrote:
                        is_first_valid_minimum = False


        if is_first_valid_minimum:
            print("Processing complete. No clean minima were found that met all criteria.")
        else:
            print(f"Successfully processed the file and saved the results to {output_file}")


    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except KeyError as e:
        print(f"Error: The input file is missing a required column: {e}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Find clean variable-star minima from an AstroImageJ measurements CSV "
            "using a global-median threshold and a symmetric odd-length window."
        ),
        usage=(
            "%(prog)s <input_file.csv> {land,space} <sequence_length> "
            "[--output OUTPUT_FILE] [--monotonic-length N] [--night-gap-hours HOURS]"
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
        night_gap_hours=args.night_gap_hours
    )
