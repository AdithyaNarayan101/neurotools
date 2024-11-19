'''
Commonly used functions 

'''

import pandas as pd
import numpy as np
import itertools
import scipy.signal as signal


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
