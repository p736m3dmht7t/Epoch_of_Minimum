import pandas as pd
import numpy as np

def find_variable_star_minima(
    input_file,
    output_file,
    sequence_length=9,
    monotonic_length=3,
    even_spacing_tolerance=600
):
    """
    Reads an AstroImageJ Measurements file and finds clean, variable star minima.

    A 'clean' minimum must satisfy three conditions:
    1. It must have a specific number of observations before and after the peak magnitude.
    2. The time stamps (BJD_TDB) of these observations must be approximately evenly spaced.
    3. The magnitudes leading into the minimum must be monotonically increasing, and the
       magnitudes leading out must be monotonically decreasing.

    Args:
        input_file (str): Path to the AstroImageJ Measurements file.
        output_file (str): Path to the output CSV file.
        sequence_length (int): The total number of data points for a valid minimum. Must be odd.
        monotonic_length (int): The number of points at the start and end of the sequence
                                to check for monotonic behavior.
        even_spacing_tolerance (int): The maximum allowed time difference in seconds from the
                                      median observation gap.
    """
    if sequence_length % 2 == 0:
        print("Error: SEQUENCE_LENGTH must be an odd number.")
        return

    half_sequence = sequence_length // 2

    try:
        # Read and prepare the data
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip()

        # Ensure required columns are numeric and drop rows with invalid data
        for col in ['BJD_TDB', 'Source_AMag_T1']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['BJD_TDB', 'Source_AMag_T1'], inplace=True)


        # Sort by time to ensure chronological order before processing
        df.sort_values(by='BJD_TDB', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Find all potential minima (local maxima in magnitude)
        # A point is a peak if it's greater than its two neighbors
        candidate_indices = np.where(
            (df['Source_AMag_T1'].shift(1) < df['Source_AMag_T1']) &
            (df['Source_AMag_T1'].shift(-1) < df['Source_AMag_T1'])
        )[0]
        
        # Open the output file to write validated minima
        with open(output_file, 'w', newline='') as csvfile:
            is_first_valid_minimum = True

            # Loop through each candidate and validate it
            for index in candidate_indices:
                # 1. BOUNDARY CHECK: Ensure there are enough data points before and after
                if not (index >= half_sequence and index < len(df) - half_sequence):
                    continue

                # Extract the full potential sequence around the minimum
                sequence_df = df.iloc[index - half_sequence : index + half_sequence + 1]

                # --- VALIDATION CHECKS ---

                # 2. EVEN SPACING CHECK
                time_diffs = sequence_df['BJD_TDB'].diff().dropna() * 86400  # Convert days to seconds
                median_diff = time_diffs.median()
                max_deviation = (time_diffs - median_diff).abs().max()

                if max_deviation > even_spacing_tolerance:
                    continue  # Fails spacing check, move to next candidate

                # 3. MONOTONICITY CHECK
                mags = sequence_df['Source_AMag_T1']
                # Check if the first N points are monotonically increasing in magnitude
                is_increasing = mags.iloc[:monotonic_length].is_monotonic_increasing
                # Check if the last N points are monotonically decreasing in magnitude
                is_decreasing = mags.iloc[-monotonic_length:].is_monotonic_decreasing

                if not (is_increasing and is_decreasing):
                    continue  # Fails monotonicity check, move to next candidate

                # --- IF ALL CHECKS PASS, WRITE TO FILE ---

                # Add a blank line separator if this is not the first valid minimum found
                if not is_first_valid_minimum:
                    csvfile.write('\n')

                # Select the final columns for the output
                output_df = sequence_df[['BJD_TDB', 'Source_AMag_T1']]
                output_df.to_csv(csvfile, index=False, header=is_first_valid_minimum)
                
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

# Call the function with the dummy input and a desired output file name
find_variable_star_minima("ASAS_J221652+2229_6_B_Table_cleaned.csv", "ASAS_J221652+2229_6_B_Table_minima.csv")