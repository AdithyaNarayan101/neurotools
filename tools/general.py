'''
Commonly used functions 

'''

import pandas as pd
import numpy as np
import itertools
import scipy.signal as signal

def convert_to_0_to_360(theta_list):
    # Convert each angle in the list
    return [theta + 360 if theta < 0 else theta for theta in theta_list]
def np_sind(degrees):
    return np.sin(np.radians(degrees))

def np_cosd(degrees):
    return np.cos(np.radians(degrees))

def standard_error(x):
    return x.std() / (len(x) ** 0.5)

def jackknife_standard_error(x):
    return x.std() * (len(x) ** 0.5)


def call_object_func(obj, func_name, *args, **kwargs):
    # Use getattr to call the method of the object
    func = getattr(obj, func_name)
    return func(*args, **kwargs)

def round_array_to_nearest_5(arr):
    return np.round(arr / 5) * 5


def get_unique_labels(df,field_name):
    # Function to get unique values of df[field_name]
    unique_labels=np.sort(df[field_name].unique())
    unique_labels=pd.Series(unique_labels).dropna().tolist() # remove any nans
    return unique_labels


def assign_bins(numbers, min_val, max_val, bin_size):
    """
    Assigns bin labels to a list of float numbers based on specified bin parameters.

    Parameters:
    - numbers (list of float): A list of float numbers to be binned.
    - min_val (float): The minimum value defining the start of the bins.
    - max_val (float): The maximum value defining the end of the bins.
    - bin_size (float): The size of each bin.

    Returns:
    - bin_labels: A list of integers representing the bin index for each number.
            Returns None for numbers outside the specified range.
    - bin_mid: A list of mid-points for each bin.    
    
    """
    # Calculate the number of bins
    num_bins = int((max_val - min_val) / bin_size)
    
    # Initialize a list to store bin labels
    bin_labels = []
    bin_mid=[]
    for number in numbers:
        
        if np.isnan(number) or number < min_val or number >= max_val:
            # If the number is outside the range, append None or a specific label
            bin_labels.append(None)
            bin_mid.append(None)
        else:
            # Calculate the bin index
            bin_index = int((number - min_val) / bin_size)
            bin_labels.append(bin_index)
            bin_mid.append(min_val + (bin_index + 0.5) * bin_size)
    
    return bin_labels, bin_mid


def mean_by_condition(df, metric_info, conditions):
        
        """
        Create a DataFrame with mean of a column (e.g. Kinematics like RT, Peak Speed, Peak Speed Time) 
        for each combination of condition values.
        
        Input: 
            metric_info: Name of the column (metric) to be computed (e.g. 'Reaction Time')
            conditions: Pairs of key-value arguments. Each key is a column,
                            each value is a list of values that column can take
                            e.g. {'Reward Size'=['Small','Large']}
            
        Output:
            df with mean metrics (rows) by conditions (columns)
        
        """
        if metric_info not in df.columns:
            raise Exception("Metric does not exist in dataframe")
            
        results = []
        
        # Create combination of values for each condition
        for value_combination in itertools.product(*conditions.values()):
            condition_dict = dict(zip(conditions.keys(), value_combination)) # Dict with each combination of condition-values
            filtered_df=df
            for condition, value in condition_dict.items():
                if isinstance(value, list):
                    
                    filtered_df = filtered_df[filtered_df[condition].isin(value)].reset_index(drop=True)
                else:
                    filtered_df = filtered_df[filtered_df[condition] == value].reset_index(drop=True)
            mean_col_value = np.nanmean(filtered_df[metric_info])
            
            # Append results to the list
            results.append({**condition_dict, 'Value': mean_col_value})
            
        # Create a DataFrame from the results
        result_df = pd.DataFrame(results)
        
        return result_df
    


def resample_signal(original_signal, original_fs, new_fs):
    """
    Resample a signal from an original sampling frequency (original_fs) to a new sampling frequency (new_fs).
    
    Parameters:
        original_signal (numpy.ndarray): The input signal to be resampled.
        original_fs (float): The original sampling frequency (Hz).
        new_fs (float): The target sampling frequency (Hz).
        
    Returns:
        numpy.ndarray: The resampled signal.
        numpy.ndarray: The new time vector corresponding to the resampled signal.
    """
    # Calculate the resampling factor
    resample_factor = new_fs / original_fs
    
    # Generate the new time axis
    t_original = np.arange(0, len(original_signal)) / original_fs
    t_new = np.arange(0, len(original_signal) * resample_factor) / new_fs
    
    # Resample the signal using scipy.signal.resample
    resampled_signal = signal.resample(original_signal, len(t_new))
    
    return resampled_signal, t_new


# def subselect_trials_idx(df, conditions):
#     """
#     Subselect trials from the DataFrame based on conditions.

#     Parameters:
#     - df (pd.DataFrame): The DataFrame containing trial data with rows as trials and columns as various variables.
#     - conditions (dict): A dictionary where keys are column names (field names) and values are lists of conditions.
    
#     Returns:
#     - idx (pd.Index): The index of rows (trials) that satisfy all conditions.
#     """
#     # Initialize a boolean mask that selects all rows initially (True for all)
#     mask = pd.Series([True] * len(df), index=df.index)
    
#     # Loop through each condition and update the mask
#     for field_name, possible_conditions in conditions.items():
#         # Ensure the field_name exists in the DataFrame
#         if field_name in df.columns:
#             # Update the mask: select only rows where the field_value is in the list of conditions
#             mask &= df[field_name].isin(possible_conditions)
#         else:
#             raise ValueError(f"Field '{field_name}' not found in the DataFrame.")
    
#     # Return the index of rows that satisfy all conditions
#     return df[mask].index


def subselect_trials_idx(df, conditions, return_indices=False):
    """
    Subselect trials from the DataFrame based on conditions.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing trial data with rows as trials and columns as various variables.
    - conditions (dict): A dictionary where keys are column names (field names) and values are lists of conditions.
    - return_indices (bool): Whether to return the indices of the selected trials (default is False).
    
    Returns:
    - tuple or pd.DataFrame: 
      - If `return_indices=True`, returns a tuple with the subselected DataFrame and the index of rows (trials) that satisfy all conditions.
      - If `return_indices=False`, returns the subset DataFrame of the trials that satisfy all conditions.
    """
    # Initialize a boolean mask that selects all rows initially (True for all)
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Loop through each condition and update the mask
    for field_name, possible_conditions in conditions.items():
        # Ensure the field_name exists in the DataFrame
        if field_name in df.columns:
            # Update the mask: select only rows where the field_value is in the list of conditions
            mask &= df[field_name].isin(possible_conditions)
        else:
            raise ValueError(f"Field '{field_name}' not found in the DataFrame.")
    
    # If return_indices is True, return both the DataFrame and the indices of the selected rows
    if return_indices:
        return df[mask].reset_index(drop=True), df[mask].index
    else:
        # Otherwise, return the subset DataFrame
        return df[mask].reset_index(drop=True)



def subsample_balance_classes(data, labels, return_indices=False):
    # Ensure that the input labels are a 1D array
    labels = np.array(labels)
    
    # Get unique classes and their indices
    unique_classes = np.unique(labels)
    
    # Initialize lists to collect subsampled data, labels, and indices
    subsampled_data = []
    subsampled_labels = []
    subsampled_indices = []
    
    # Find the minimum count of any class in the labels
    min_class_count = min(np.sum(labels == c) for c in unique_classes)
    
    # Loop over each class and subsample
    for c in unique_classes:
        # Get indices of the current class
        class_indices = np.where(labels == c)[0]
        
        # Randomly sample 'min_class_count' samples from the current class
        sampled_indices_class = np.random.choice(class_indices, min_class_count, replace=False)
        
        # Add the corresponding data, labels, and indices to the subsample lists
        subsampled_data.append(data[sampled_indices_class])
        subsampled_labels.append(labels[sampled_indices_class])
        subsampled_indices.append(sampled_indices_class)
    
    # Concatenate the subsampled data, labels, and indices
    subsampled_data = np.concatenate(subsampled_data, axis=0)
    subsampled_labels = np.concatenate(subsampled_labels, axis=0)
    subsampled_indices = np.concatenate(subsampled_indices, axis=0)
    
    # Return the results
    if return_indices:
        return subsampled_data, subsampled_labels, subsampled_indices
    else:
        return subsampled_data, subsampled_labels
    

def subsample_trials_df(df, N, return_indices=False):
    """
    Randomly subsample N trials from the rows of a Pandas DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame where rows represent trials.
    - N (int): The number of trials to subsample.
    - return_indices (bool): If True, also return the indices of the selected rows.

    Returns:
    - If return_indices is False, returns the subsampled DataFrame (pandas.DataFrame).
    - If return_indices is True, returns a tuple (subsampled_df, indices).
    """
    # Check if DataFrame has more than N trials (rows)
    if df.shape[0] < N:
        raise ValueError(f"Input DataFrame must have more than {N} trials (rows).")
    
    # Randomly sample N rows from the DataFrame
    subsampled_df = df.sample(n=N, random_state=42)
    
    # Return subsampled DataFrame and optionally the indices
    if return_indices:
        return subsampled_df.reset_index(drop=True), subsampled_df.index.to_numpy()  # Return indices as NumPy array
    else:
        return subsampled_df.reset_index(drop=True)
    
def index_blocks(sequence):
    indexed_sequence = []
    current_value = sequence[0]  # Start with the first value in the sequence
    block_num = 1  # Start with block 1

    # Iterate through the sequence and index blocks of 1's and 2's
    i = 0
    while i < len(sequence):
        block = []
        
        # Collect the current block of identical numbers (1 or 2)
        while i < len(sequence) and sequence[i] == current_value:
            block.append(block_num)  # Assign the block number
            i += 1

        indexed_sequence.extend(block)  # Add the indexed block to the sequence
        block_num += 1  # Increment block number for the next block

        # Alternate the current value between 1 and 2
        current_value = 2 if current_value == 1 else 1

    return indexed_sequence


