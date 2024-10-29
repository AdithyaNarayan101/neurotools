'''
Module to analyze behavior

- Compute basic behavioral metrics like mean success rates, hit rates, false alarm rates, motor accuracy 

'''
import itertools
import pandas as pd
import numpy as np

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
    
    
    
    
        
    


        
     
        
        
    
        
    
    
    
    
        

        
        
        
        