import sys
import os
import pandas as pd
import numpy as np

from kwee_and_van_woerden import kwee_van_woerden


def _trim_and_validate_sequence(
    sequence_df,
    *,
    sequence_length,
    monotonic_length
):
    working_df = sequence_df.copy().reset_index(drop=True)
    if working_df.empty:
        return None

    first_mag = working_df['Source_AMag_T1'].iloc[0]
    last_mag = working_df['Source_AMag_T1'].iloc[-1]

    trimmed_df = working_df.copy()
    if first_mag >= last_mag:
        while len(trimmed_df) > 1 and trimmed_df['Source_AMag_T1'].iloc[-1] < first_mag:
            trimmed_df = trimmed_df.iloc[:-1]
    else:
        while len(trimmed_df) > 1 and trimmed_df['Source_AMag_T1'].iloc[0] < last_mag:
            trimmed_df = trimmed_df.iloc[1:]

    if trimmed_df.empty:
        trimmed_df = working_df

    orig_diff = abs(first_mag - last_mag) if len(working_df) > 1 else 0.0
    trimmed_first = trimmed_df['Source_AMag_T1'].iloc[0]
    trimmed_last = trimmed_df['Source_AMag_T1'].iloc[-1]
    trimmed_diff = abs(trimmed_first - trimmed_last) if len(trimmed_df) > 1 else 0.0

    candidate_df = working_df if trimmed_diff > orig_diff else trimmed_df
    if len(candidate_df) < sequence_length:
        return None
    candidate_df = candidate_df.reset_index(drop=True)

    if monotonic_length > 0:
        if len(candidate_df) < monotonic_length:
            return None
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
    sequence_length=40,
    monotonic_length=10,
    night_mode='space',
    night_gap_hours=8.0
):
    """
    Reads an AstroImageJ Measurements file and finds clean, variable star minima.

    A 'clean' minimum must satisfy three conditions:
    1. It belongs to a contiguous sequence of observations with magnitudes above the
       dataset median within a nightly segment.
    2. After trimming to balance the sequence ends, it contains at least `sequence_length`
       samples centered on the maximum magnitude.
    3. The first `monotonic_length` magnitudes are non-decreasing and the last
       `monotonic_length` are non-increasing.

    Args:
        input_file (str): Path to the AstroImageJ Measurements file.
        output_file (str): Path to the output CSV file.
        sequence_length (int): The total number of data points for a valid minimum. Must be odd.
        monotonic_length (int): The number of points at the start and end of the sequence
                                to check for monotonic behavior.
        night_mode (str): 'land' splits nights by gaps exceeding `night_gap_hours`; 'space'
                          treats the light curve as a single continuous segment.
        night_gap_hours (float): Time gap (in hours) that defines separate nights in land mode.
    """
    try:
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

        if night_mode_normalized == 'land':
            if night_gap_hours <= 0:
                raise ValueError("night_gap_hours must be positive for land mode.")
            gap_days = night_gap_hours / 24.0
            diffs = df['BJD_TDB'].diff().fillna(0)
            df['segment'] = (diffs > gap_days).cumsum()
        else:
            df['segment'] = 0

        # Open the output file to write validated minima
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide an input file name.")
        print("Usage: python find_minima.py <input_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Generate output filename by appending '_minima' before the .csv extension
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_minima.csv"
    
    find_variable_star_minima(input_file, output_file)