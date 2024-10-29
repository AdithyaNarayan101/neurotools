'''
Commonly used functions 

'''

import pandas as pd
import numpy as np
import itertools
def standard_error(x):
    return x.std() / (len(x) ** 0.5)

def jackknife_standard_error(x):
    return x.std() * (len(x) ** 0.5)


def call_object_func(obj, func_name, *args, **kwargs):
    # Use getattr to call the method of the object
    func = getattr(obj, func_name)
    return func(*args, **kwargs)


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