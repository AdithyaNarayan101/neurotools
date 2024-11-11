'''
Module to analyze behavior

- Compute basic behavioral metrics like mean success rates, hit rates, false alarm rates, motor accuracy 

'''
import itertools
import pandas as pd
import numpy as np
from scipy.stats import norm
class behavior_metrics:
    ''' 
    Class for behavioral metrics. 
    
    Inputs: 
        Dataframe: pd.dataFrame (see below for more details)
        result_name: string specifying the name of the column with the results of each trial (e.g. "Result")
    
    Dataframe
        Rows: Trials 
        Columns:
            Conditions (e.g. Reaward Size)
            Result (string, e.g. CORRECT)
            Reaction Time / Peak Velocity (float)
    
    '''
    
    def __init__(self, df, result_name="Result"):
        self.df = df
        self.result=result_name
        
    def filter_rows(self, column, value):
        """Filter the DataFrame based on a column value."""
        filtered_df = self.df[self.df[column] == value]
        return filtered_df
        
        
    def outcome_rate_subselect_trials(self, numerator_outcomes, denominator_outcomes, conditions):
        """Compute outcome rate using specified numerator and denominator outcomes."""
        filtered_df = self.df
        
        for condition, value in conditions.items():
            filtered_df = filtered_df[filtered_df[condition] == value]
            
        # Count the occurrences of the numerator outcomes
        counts_numerator = filtered_df[self.result].isin(numerator_outcomes).sum()
        
        # Count the occurrences of the denominator outcomes
        counts_denominator = filtered_df[self.result].isin(denominator_outcomes).sum()
        
        # Calculate rate, avoiding division by zero
        rate = counts_numerator / counts_denominator if counts_denominator > 0 else np.nan
        
        return rate*100

    def outcome_rate_by_conditions(self, conditions, metric_info):
        
        """Create a DataFrame with outcome rates for each combination of condition values.
            Input: 
                conditions: Pairs of key-value arguments. Each key is a column,
                            each value is a list of values that column can take
                            e.g. {'Reward Size'=['Small','Large']}
                metric_info: dict with two entries
                    numerator : List of outcomes in the numerator 
                    denominator: List of outcomes in the denominiator
                
                        
            Output:
                 df with outcome rates (rows) by conditions (columns)
                 
        """
        
        results = []
    
        # Create combinations of values for each condition
        for value_combination in itertools.product(*conditions.values()):
            condition_dict = dict(zip(conditions.keys(), value_combination)) # Dict with each combination of condition-values
            # Compute the outcome rate for this combination
            rate = self.outcome_rate_subselect_trials(metric_info['numerator'], metric_info['denominator'], condition_dict)
            # Append results to the list
            results.append({**condition_dict, 'Value': rate})
            
        # Create a DataFrame from the results
        result_df = pd.DataFrame(results)
        
        return result_df
    
   
    def SDT_metris_by_condition(self, conditions, metric_info):
        
        """Create a DataFrame with signal detection metrics (dPrime,Bias) for each combination of condition values.
            Input: 
                conditions: Pairs of key-value arguments. Each key is a column,
                            each value is a list of values that column can take
                            e.g. {'Reward Size'=['Small','Large']}
                metric_info: dict with two entries
                    metric_name: name of metric to be computed (e.g. dPrime, Bias)
                    outcome_names: Dict with field-names for outcomes used in SDT modelling ('HIT':['CORRECT','MOTOR_ERROR'])
                        
            Output:
                 df with SDT metric (rows) by conditions (columns)
                 
        """
        results = []
        
        # Create combinations of values for each condition
        for value_combination in itertools.product(*conditions.values()):
            condition_dict = dict(zip(conditions.keys(), value_combination)) # Dict with each combination of condition-values
            
            
#             value = self.compute_SDT_metrics(metric_info['numerator'], metric_info['denominator'], condition_dict)
            
            filtered_df=self.df
            for condition, value in condition_dict.items():
                filtered_df = filtered_df[filtered_df[condition] == value]
                
            sdt_metrics = compute_SDT_metrics(filtered_df, outcome_names = metric_info['outcome_names'])
            # Append results to the list
            results.append({**condition_dict, 'Value': sdt_metrics[metric_info['metric_name']]})
            
        # Create a DataFrame from the results
        result_df = pd.DataFrame(results)
        
        return result_df
        
# Functions for modifying df to make it easier for behavioral analyses:

import pandas as pd
import time
def create_correct_rejects(df,  delay_bin_idx_name = 'Delay Bin Idx', delay_bin_name = 'Delay Bin'):
    """
    Modify the DataFrame by creating duplicates for trials with specific results.

    For each trial in the DataFrame:
    - If the 'Result' is in the list of hit_trials, create N-1 duplicates,
      where N is the value in 'Delay Bin'. Each duplicate will have 'Result' 
      set to "CORRECT REJECT" and 'Delay Bin' set to i (1 to N-1).
      
    - If the 'Result' is in the list of fa_trials, create M-1 duplicates,
      where M is the value in 'Go Time'. Each duplicate will again have 
      'Result' set to "CORRECT REJECT" and 'Delay Bin' set to i (1 to M-1).

    Parameters:
    df (pd.DataFrame): The input DataFrame containing trial data with columns 
                       'Result', delay_bin_idx_name , and 'Go Time'.
   

    Returns:
    pd.DataFrame: A new DataFrame with the original rows and the added duplicates 
                  for the specified trials.
    """
    
    # List to store new rows
    new_rows = []
    start_time=time.time()
    
    ## Get unique delay value (dict with idx as keys and bin-mid (ms) as values)
    # Generate pairs and create a dictionary 
    unique_pairs = df.apply(lambda row: (row[delay_bin_idx_name], row[delay_bin_name]), axis=1).drop_duplicates()
    # Convert pairs to a dictionary (col1 as key, col2 as value)
    unique_delays_dict = dict(unique_pairs.tolist())
    
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Check if the Result is in hit_trials
        try:
            N = int(row[delay_bin_idx_name])
        except:
            continue
        # Create N-1 duplicates for each delay bin before the current bin
        for i in range(0, N-1):
            new_row = row.copy()
            new_row['Result'] = "CORRECT_REJECT"
            new_row[delay_bin_idx_name] = i  # Assigns delay idx for this CR trial as i
            try:
                new_row[delay_bin_name]=unique_delays_dict[i]  # Assigns delay bin for this CR trial
            except:
                
                # For some really long trials, certain bins might not have occurred at all 
                # We don't analyze time bins longer than 2000ms usually, so fine to ignore it. 
                print ("Couldn't find corresponding Delay Bin value") # Print warning message - if this happens often, worth looking into why!
                new_row[delay_bin_name]=np.nan 
                
            # Append Correct Reject trial    
            new_rows.append(new_row)
    
        if index%5000==0:
            end_time=time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds, trial " + str(index))
            start_time = time.time()
        
    # Concatenate the new rows to the original DataFrame
    new_rows_df=pd.DataFrame(new_rows)
    new_df = pd.concat([df,new_rows_df], ignore_index=True)
    return new_df



def calculate_delay_times(go_times, reaction_time_mean, change_times, response_window, flag):
    """
    Calculate delay times based on go times, reaction time mean, and change times.

    Parameters:
    go_times (list): A list of go times for trials.
    reaction_time_mean (float): The mean reaction time to be used for delay calculation.
    change_times (list): A list of change times for the trials.
    response_window (tuple): A tuple containing two values that define the response window.
                             The first value is the lower bound, and the second value is the upper bound.
    flag (str): A flag that determines the calculation method. 
                Options are 'bin by reaction time'  or 'bin by change time'

    Returns:
    list: A list of calculated delay times.
    """
    
    delay_times = []
    
    if flag == 'bin by reaction time':
        # Calculate delay times by subtracting the reaction time mean from go times
        delay_times = [go_time - reaction_time_mean for go_time in go_times]
    
    elif flag == 'bin by change time':
        # Calculate delay times based on change times and response window
        for go_time in go_times:
            delay_found = False
            for i, change_time in enumerate(change_times):
                lower_bound = change_time + response_window[0]
                upper_bound = change_time + response_window[1]
                
                if lower_bound <= go_time <= upper_bound:
                    delay_times.append(change_time)
                    delay_found = True
                    break
            
            # If no matching change time is found, append None or handle as necessary
            if not delay_found:
                delay_times.append(np.nan)  # or some other placeholder value
    
    return delay_times



def compute_SDT_metrics(df, outcome_names):
    """
    Function to compute signal detection theory measures of dPrime, criterion, and Bias
    
    See Stanislaw, Todorov, 1999 or Macmillan & Creelman Detection Theory: a user's guide
    
    """
    # create a new field with results re-labelled to match SDT conventions
    df['SDT Result'] = df['Result']
    df['SDT Result'].loc[df['Result'].isin(outcome_names['Hit'])]  = 'Hit'
    df['SDT Result'].loc[df['Result'].isin(outcome_names['Miss'])]  = 'Miss'
    df['SDT Result'].loc[df['Result'].isin(outcome_names['CR'])]  = 'CR'
    df['SDT Result'].loc[df['Result'].isin(outcome_names['FA'])]  = 'FA'
    
    # Count the occurrences of each trial result
    counts = df['SDT Result'].value_counts()

    # Extract the number of Hits, Misses, False Alarms, and Correct Rejects
    hit_count = counts.get('Hit', 0)
    miss_count = counts.get('Miss', 0)
    false_alarm_count = counts.get('FA', 0)
    correct_reject_count = counts.get('CR', 0)
    
    # Total number of trials
    total_trials = len(df)
    
    # Calculate Hit Rate (HR) and False Alarm Rate (FAR)
    hit_rate = hit_count / (hit_count + miss_count) if (hit_count + miss_count) > 0 else np.nan
    false_alarm_rate = false_alarm_count / (false_alarm_count + correct_reject_count) if (false_alarm_count + correct_reject_count) > 0 else np.nan
    
    # Apply Z-transform to HR and FAR
    # Ensure no zero rates for Z-transform (use 0.01 and 0.99 as safe bounds for small probabilities)
    hit_rate = np.clip(hit_rate, 0.01, 0.99)
    false_alarm_rate = np.clip(false_alarm_rate, 0.01, 0.99)
    
    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(false_alarm_rate)
    
    # Compute d' (d-prime)
    d_prime = z_hit - z_fa
    
    # Compute Criterion (C) as (z_hit + z_fa) / 2
    criterion = -(z_hit + z_fa) / 2
    
    # Compute Response Bias (ln(beta))
    bias = (z_fa**2 - z_hit**2)/2
    
    
    # Return results as a dictionary
    return {
        'dPrime': d_prime,
        'criterion': criterion,
        'bias': bias
    }


     
        
        
    
        
    
    
    
    
        

        
        
        
        