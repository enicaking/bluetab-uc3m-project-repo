# ================================================================
# Universidad Carlos III de Madrid · Bluetab Collaboration Project
# ---------------------------------------------------------------
# preprocessing.py
# ---------------------------------------------------------------
# Description:
#   Deterministic preprocessing pipeline for the Bluetab fraud project.
#   This module centralizes all data-cleaning logic to ensure that every
#   notebook and modeling script uses the exact same transformations.
#
#   The pipeline performs:
#     - Duplicate removal and index resetting in all source tables
#     - Manual removal of known inconsistent (transaction_id, customer_id) pairs
#     - Deterministic merging of all datasets:
#           • INNER JOIN on transaction_id for transactions/locations/flags/time
#           • INNER JOIN on device_id for device metadata
#           • LEFT JOIN on customer_id to retain all valid transactions
#     - Light schema normalization (column renames, safe type casting)
#     - Missing-values handling based on EDA decisions:
#           • Drop rows with missing customer information ("name")
#             since EDA confirmed none of them contain fraud
#           • Fill sparse categorical/contact fields with "Unknown"
#
#   No visualizations or EDA logic are included here.
#   This module is meant to be imported by notebooks or modeling scripts
#   to guarantee reproducibility and consistency across the project.
#
# Usage example:
#     import sys
#     from pathlib import Path
#
#     # Go from /notebooks to project root
#     repo_root = Path.cwd().parent
#     sys.path.append(str(repo_root))
#
#     from bluetab_fraud.preprocessing import pipeline_preprocessing
#     df = pipeline_preprocessing(
#         transactions_df, locations_df, customers_df,
#         flags_df, time_table_df, devices_df, verbose=True
#     )
#
# ================================================================


import pandas as pd

def drop_and_reset(df: pd.DataFrame, name: str, verbose: bool = False) -> pd.DataFrame:
    """Drop duplicate rows and reset the index of a DataFrame.

    Parameters:
    - df: DataFrame to process
    - name: descriptive name used in messages when verbose=True
    - verbose: if True, print row counts before and after processing

    Returns the modified DataFrame (in-place mutation is performed and the
    same DataFrame is also returned for convenience).
    """
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    after = len(df)
    if verbose:
        print(f"{name}: {before} -> {after} rows (duplicates removed: {before-after})")
    return df

def merge_all_tables( transactions_df, locations_df, flags_df, time_table_df, devices_df, customers_df):
    """
    Merge all project tables into a single unified dataframe.

    The merge logic follows the project’s relational structure:
    - transactions ↔ locations  (1-to-1, key: transaction_id)
    - + flags                   (1-to-1, key: transaction_id)
    - + time_table             (1-to-1, key: transaction_id)
    - + devices                (many-to-1, key: device_id)
    - + customers              (many-to-1, key: customer_id)

    Notes
    -----
    * All merges except the customers table use INNER JOIN to ensure
      transactional consistency (we only keep fully valid records).
    * The final merge with customers uses LEFT JOIN because customer 
      information may be missing for some device or transaction records.

    Returns
    -------
    df : pandas.DataFrame
        Final merged dataset ready for preprocessing steps.
    """

    # 1. transactions + locations
    merge1 = pd.merge(
        transactions_df,
        locations_df,
        on="transaction_id",
        how="inner"
    )

    # 2. + flags
    merge2 = pd.merge(
        merge1,
        flags_df,
        on="transaction_id",
        how="inner"
    )

    # 3. + time table (temporal metadata)
    merge3= pd.merge(
        merge2,
        time_table_df,
        on="transaction_id",
        how="inner"
    )

    # 4. + devices (device metadata, device_id is FK)
    merge4 = pd.merge(
        merge3,
        devices_df,
        on="device_id",
        how="inner"
    )

    # 5. + customers (demographics; LEFT JOIN to avoid losing transactions)
    df = pd.merge(
        merge4,
        customers_df,
        on="customer_id",
        how="left"
    )

    return df


def pipeline_preprocessing(transactions_df, locations_df, customers_df, flags_df, time_table_df, devices_df, verbose: bool = False):
    """Perform common preprocessing steps before merging datasets.

    Currently this function performs the following actions:
    - Removes duplicate rows from each input DataFrame
    - Resets indices to ensure contiguous integer indexes

    Parameters:
    - transactions_df, locations_df, customers_df, flags_df, time_table_df, devices_df: DataFrames to clean
    - verbose: if True, print row counts before and after for each DataFrame

    Returns a tuple with the cleaned DataFrames in the same order.
    """
    # Remove rows duplicated in all the datasets
    transactions_df = drop_and_reset(transactions_df, "transactions_df", verbose=verbose)
    locations_df = drop_and_reset(locations_df, "locations_df", verbose=verbose)
    customers_df = drop_and_reset(customers_df, "customers_df", verbose=verbose)
    flags_df = drop_and_reset(flags_df, "flags_df", verbose=verbose)
    time_table_df = drop_and_reset(time_table_df, "time_table_df", verbose=verbose)
    devices_df = drop_and_reset(devices_df, "devices_df", verbose=verbose)

    # Remove the rows in transaction that are incorrect because the customer_id does not match in the transaction database with the customer database
    bad_pairs = [
            ("a995c6a8-ef9d-4c4f-928d-7149a5549fc8", 99180),
            ("70a09c87-2693-4455-9373-01c07f4cbc65", 99172),
            ("7dd260b9-5836-4d26-9163-ceff19cee458", 99209),
        ]
    for tx, cust in bad_pairs:
        if {"transaction_id", "customer_id"}.issubset(transactions_df.columns):
            before = len(transactions_df)
            transactions_df = transactions_df[~(
                (transactions_df["transaction_id"] == tx) &
                (transactions_df["customer_id"]   == cust)
            )]
            after = len(transactions_df)
            if verbose and before != after:
                print(f"[Clean] Removed 1 inconsistent row: (transaction_id={tx}, customer_id={cust}).")

    # Merge all databases
    df = merge_all_tables(transactions_df=transactions_df, locations_df=locations_df, customers_df=customers_df, flags_df=flags_df,time_table_df=time_table_df,devices_df=devices_df)

    # Rename country variables
    if "country_x" in df.columns:
        df.rename(columns={'country_x': 'merchant_country'}, inplace=True)
    if "country_y" in df.columns:
        df.rename(columns={'country_y': 'customer_country'}, inplace=True)

    # Convert numeric variables into objects
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype("object")
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype("object")

    # Missing Values

    if "name" in df.columns:
        before = len(df)
        df = df.dropna(subset=["name"])
        after = len(df)
        if verbose:
            print(f"[Missing] Removed {before - after} rows without customer information (safe to drop).")

    # Fill sparse categorical/contact fields with 'Unknown'
    fill_map = {
        "zip_code": "Unknown",
        "browser": "Unknown",
        "email": "Unknown",
        "phone": "Unknown"
    }
    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
            if verbose:
                print(f"[Missing] Filled missing '{col}' with 'Unknown'.")

    return df

