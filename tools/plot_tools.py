'''
Module for plotting functions 

'''

# Import stuff
import numpy as np
import scipy as sp
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from tools.general import *


"""
Define color maps
"""

# Get a circular color map (for circular variables like orientation)
def generate_color_circle(start=0, end=360, step=5):
    color_dict = {}
    
    for hue in range(start, end+1, step):  # Include end by using end + 1
        # Ensure hue wraps around in circular space
        wrapped_hue = hue % 360
        
        # Convert hue to RGB using matplotlib
        rgb = plt.cm.hsv(wrapped_hue / 360)[:3]  # Get RGB and ignore alpha channel
        # Convert to 0-255 range and create a tuple
        rgb_255 = tuple(int(c * 255) for c in rgb)
        
        # Convert RGB to hex format
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_255)
        
        color_dict[wrapped_hue] = hex_color
    
    return color_dict


def get_color_map():
    color_circle=generate_color_circle(start=0,end=360,step=5)
    color_map={
        'Reward Size Classic':{'Small':'Red','Medium':'Orange','Large':'Blue'},
        'Reward Size':{'Small':'pink','Medium':'red','Large':'darkred'},
        'Reward Size Hex':{'Small':'#FFC0CB','Medium':'#FF0000','Large':'#8B0000'},
        'Focus Block Hex': {'Trivial':'#000000','Easy':'#0096FF','Hard':'#00008B'},
        'Focus Block': {'Trivial':'black','Easy':'lightblue','Hard':'darkblue'},
        'Target Size': {'Tiny':'darkgreen','Huge':'lightgreen'},
        'Target Size Hex': {'Tiny':'#023020','Huge':'#90EE90'},
        'Change Idx Hex': {1: '#00008B', 2: '#0000CD', 3: '#0000FF', 4: '#4169E1', 5: '#6495ED', 6: '#87CEFA', 7: '#00BFFF', 8: '#1E90FF', 9: '#00CED1', 10: '#20B2AA', 11: '#3CB371'}

,

        'Target Angle Hex':color_circle
    }

    return color_map

def get_dash_map():
    color_map={
        'Reward Size':{'Small':'dot','Medium':'dash','Large':'solid'},
        'Focus Block': {'Trivial':'dot','Easy':'dash','Hard':'solid'},
        'Target Size': {'Tiny':'dash','Huge':'solid'},
    }

    return color_map


def generate_color_list(num_colors, cmap_name = 'turbo', output_format = 'hex'):
    """
    Generate a list of colors using a specified colormap.

    Parameters:
    num_colors (int): The number of colors to generate.
    cmap_name (str): The name of the colormap to use.

    Returns:
    list: A list of RGB color tuples.
    """
    # Get the colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Generate colors
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    
    # Convert colors based on the output format
    if output_format == 'tuple':
        colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]
    elif output_format == 'string':
        colors = [f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {int(a * 255)})' for r, g, b, a in colors]
    elif output_format == 'hex':
        colors = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]
    
    return colors






"""
Define figure plotting functions

"""

def make_outcome_rate_figure(fig, behav_metrics, metrics, conditions, x_label, split_label = '', split_cond = '', plot_individual_sessions = True):

    '''
    Function to plot various behavioral metrics (like success rates, motor accuracy etc.) as a function of a condition (e.g. Change Magnitude). 

    Input: 
        fig: Plotly figure with subplots 
        behav_metrics: Object of class behavior_metrics (see tools.behavior)
        metrics : dict specifying all the metrics to be plotted and how to compute it (i.e. the trial outcomes to be counted in the numerator/denominator)
        conditions: dict with the condition names and unique values it can take (e.g. {'Reward Size': ['Small','Medium','Large']})
        x_label: condition-name to be used for x-axis
        split_label: condition-name being split by
        plot_individual_sessions: Boolean flag specifying whether to plot traces for individual sessions (index by 'Session') or only Mean+/-SE
        line_color: Color of Mean+/-SE line (string or hex)
    Output: 
        fig: Plotly figure with subplots 

    Dependencies:
        neurotools: behavior, general_functions

    '''
    # Get color map
    color_map=get_color_map()
    dash_map=get_dash_map()
    
    # Empty dict to store results
    results={} 

    legend_flag = True # Needed to suppress repeats of legend entries appearing for traces across subplots

    # Loop through all metrics, compute each metric split by conditions, store and plot results
    iMetric=0

    for metric in metrics:
        iMetric+=1
        results[metric]=behav_metrics.outcome_rate_by_conditions(numerator_outcomes=metrics[metric]['numerator']
                                                     ,denominator_outcomes=metrics[metric]['denominator'],
                                                    conditions=conditions)

        results[metric] = results[metric].sort_values(by=x_label)
        
        if(split_label != ''):
            
            # Subselect results only for subset of trials with condition-value specified by split_label-split_cond
            results[metric]=results[metric][results[metric][split_label]==split_cond].reset_index(drop=True)
            
            # Get line-color for the condition-value specified by split_label-split_cond
            line_color = color_map[split_label][split_cond]
            
            # dash type not specified for certain conditions
            try:
                dash_type = dash_map[split_label][split_cond]
            except: 
                dash_type = 'solid'
                
        else:
            line_color = 'black' # default line color for mean+/-SE
            dash_type = 'solid'
            
        ## Plot Mean +/- SE    

        # Get Mean and SE by condition using groupby: 
        df_summary=results[metric].groupby(x_label).agg(
            Mean=('Outcome Rate','mean'),
            SE = ('Outcome Rate',standard_error)
        ).reset_index()
        
        # Sorting results dataframe so that the order of values of x-label is preserved 
        df_summary[x_label] = pd.Categorical(df_summary[x_label], categories=conditions[x_label], ordered=True)
        df_summary = df_summary.sort_values(by=x_label)

        
        # Get colors for data points based on x-label: 
        try:
            x_colors = [color_map[x_label+' Hex'][i] for i in conditions[x_label]]
        except: 
            # If not defined in the color-map, just have data poins be black
            x_colors='black'
        
        
        # Add trace
        fig.add_trace(go.Scatter(x=df_summary[x_label],y=df_summary['Mean'],error_y=dict(array=df_summary['SE']), line=dict(width=4,color=line_color), marker = dict(size=10,color=x_colors), name = 'Mean +/- SE '+split_cond, legendgroup = 'Mean +/- SE '+split_cond, showlegend = legend_flag), row = 1,col = iMetric)

        
        ## Plot individual sessions:

        if(plot_individual_sessions):

            all_sessions = np.sort(results[metric]['Session'].unique())

            session_colors=generate_color_list(num_colors = len(all_sessions), cmap_name = 'turbo', output_format = 'hex') # Gets list of colors
            session_num = 0
            for session in all_sessions:
                results_session = results[metric][results[metric]['Session']==session].reset_index(drop=True)
                results_session=results_session.dropna(subset=['Outcome Rate'])
                fig.add_trace(go.Scatter(x = results_session[x_label],y = results_session['Outcome Rate'], mode='lines', line=dict(width=0.5, dash = dash_type, color=session_colors[session_num]), name = str(session)+' '+split_cond, legendgroup = str(session)+' '+split_cond, showlegend = legend_flag), row = 1, col = iMetric)
                session_num +=1

        legend_flag = False 
    
    # Clean up figure:
    fig.update_layout(width=2000,height=400)       
    fig['layout']['yaxis']['title']='Percentage(%)'
    fig.update_xaxes(title_text=x_label)
    
    return fig
