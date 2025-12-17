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
import plotly.colors
import matplotlib.colors as mcolors
from tools.neural_analyses import *
from scipy.stats import zscore

import plotly.io as pio

def set_ploty_defaults():
    pio.templates["my_default"] = pio.templates["plotly_white"].update({
        "layout": {
            "font": {
                "family": "Arial",
                "size": 20,        # global base font size
                "color": "black",
            },
            "title": {
                "font": {"size": 30}
            },
            "xaxis": {
                "title": {"font": {"size": 25}},
                "tickfont": {"size": 20},
            },
            "yaxis": {
                "title": {"font": {"size": 25}},
                "tickfont": {"size": 20},
            },
            "legend": {
                "font": {"size": 20}
            },
            "margin": {"l": 70, "r": 30, "t": 60, "b": 60},
        }
    })

    pio.templates.default = "my_default"



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


def generate_color_shades(color_name, num_shades=10):
    # Ensure the color name is valid
    if color_name not in mcolors.CSS4_COLORS:
        raise ValueError(f"Color name '{color_name}' is not recognized. Please use a valid CSS color name.")
    
    # Get RGB values of the color name
    base_color = mcolors.CSS4_COLORS[color_name]
    rgb = np.array(mcolors.hex2color(base_color))  # Convert to RGB (normalized [0, 1] range)
    
    # Generate the shades by adjusting the brightness
    shades = []
    num_shades=num_shades+1
    for i in range(1,num_shades):
        # Adjust brightness: interpolate between 0 (dark) and 1 (light)
        factor = 1- (i / (num_shades - 1))  # A value between 0 and 1
        shade_rgb = rgb * (1 - factor) + factor  # Blend the base color with white (lightening)
        hex_color = mcolors.rgb2hex(shade_rgb)  # Convert RGB back to hex
        shades.append(hex_color)
    
    return shades
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

Define plotting helper functions

"""
def set_axis_ranges(fig, x_range=None, y_range=None):
    """
    Set the specified x and y ranges across all subplots of the given Plotly figure.

    Parameters:
    - fig: Plotly figure object.
    - x_range: Tuple or list specifying the desired x-axis range [min, max].
    - y_range: Tuple or list specifying the desired y-axis range [min, max].

    Returns:
    - fig: Updated Plotly figure with specified axis ranges applied.
    """
    # Apply the specified x-range and y-range to all x and y axes
    for axis in fig.layout:
        if axis.startswith('xaxis') and x_range is not None:
            fig.update_layout({axis: {'range': x_range}})
        elif axis.startswith('yaxis') and y_range is not None:
            fig.update_layout({axis: {'range': y_range}})
    
    return fig
def make_bg_white(fig, rows, cols):
    """
    Update a Plotly figure to have a white background and black axes.

    Parameters:
        fig: plotly.graph_objects.Figure
            The Plotly figure to be updated.
    """
    # Set the background colors
    fig.update_layout(
        paper_bgcolor='white',  # Background color of the entire figure
        plot_bgcolor='white'    # Background color of the plotting area
    )
    
    

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            fig.update_xaxes(linecolor='black', tickcolor='black', row=row, col=col)
            fig.update_yaxes(linecolor='black', tickcolor='black', row=row, col=col)
    return fig



def hex_to_rgba(hex_color, alpha=1.0):
    """
    Convert a hex color to RGBA format with a specified alpha transparency.
    
    Parameters:
        hex_color (str): The hex color code (e.g., '#FF5733').
        alpha (float): The alpha transparency value (0.0 to 1.0).
        
    Returns:
        str: The color in 'rgba' format (e.g., 'rgba(255, 87, 51, 1.0)').
    """
    # Convert hex to RGB tuple (values between 0 and 1)
    rgb = mcolors.hex2color(hex_color)
    
    # Scale RGB values to 0-255
    r, g, b = [int(c * 255) for c in rgb]
    
    # Return the color in 'rgba' format
    return f'rgba({r}, {g}, {b}, {alpha})'


  
def stdshade(fig, data, x_values, error_type='sem', row=1, col=1, color='#FF5733', name="Mean with Shaded Error", xlabel="Time", ylabel="",legend_flag=False, line_width=1,line_dash='solid'):
    """
    Plot the mean of a set of signals with a shaded region representing the error (standard deviation or standard error).
    Add the plot to a specific subplot in the existing figure.
    
    Parameters:
        fig (plotly.graph_objects.Figure): The figure to add the subplot trace to.
        data (list of numpy.ndarray): A list of 2D numpy arrays (each array is a signal to plot).
        error_type (str): Type of error to show in the shaded region ('std' for standard deviation, 'sem' for standard error of the mean).
        row (int): The row index of the subplot.
        col (int): The column index of the subplot.
        color (str): Color for the mean and shaded region.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        legend_flag : True/False 
        
    Returns:
        None: Adds traces to the given subplot in the figure.
    """
    # Check if all signals are the same length
    signal_length = len(data[0])
    for signal in data:
        if len(signal) != signal_length:
            raise ValueError("All signals must have the same length.")
    
    # Convert data to a numpy array for easier manipulation
    data_array = np.array(data)
    
    # Calculate the mean of the signals
    mean_signal = np.nanmean(data_array, axis=0)
    
    # Calculate the error (std or sem)
    if error_type == 'std':
        error = np.nanstd(data_array, axis=0)
    elif error_type == 'sem':
        error = np.nanstd(data_array, axis=0) / np.sqrt(len(data))
    else:
        raise ValueError("Error type must be 'std' or 'sem'.")
    color_rgba = hex_to_rgba(color,alpha=.4)
    color_rgba_white = hex_to_rgba("white",alpha=0)
    # Plot the mean signal in the specified subplot
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=mean_signal, 
        mode='lines', 
        name=name, 
        legendgroup=name,
        line=dict(color=color, width=line_width,dash=line_dash),
        showlegend=legend_flag
    ), row=row, col=col)
    # Add shaded error region in the specified subplot
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_values, np.flip(x_values)]),
        y=np.concatenate([mean_signal - error, np.flip(mean_signal + error)]),
        fill='toself',
        fillcolor=color_rgba,  # Using the color with transparency for the shaded region
        line=dict(color=color_rgba_white),  # No border for the shaded region
        name='Error Region',
        legendgroup=name,
        showlegend=False
    ), row=row, col=col)
    
    # Update layout for the subplot
#     fig.update_xaxes(title_text=xlabel, row=row, col=col)
    fig.update_yaxes(title_text=ylabel, row=row, col=col)


    

"""
Define figure plotting functions

"""

def plot_metrics_by_condition(fig, data, func_compute_metric, metrics, conditions, x_label, split_label = '', split_cond = '', plot_individual_sessions = True, column_num=1, column_num_type='automatic',row_num=1, row_num_type = 'automatic',color_map='black',dash_map='solid',make_scatter_plot=False):

    '''
    Function to plot various metrics (like success rates, RT, decoding accuracy etc.) as a function of a condition (e.g. Change Magnitude)

    Input: 
        fig: Plotly figure with subplots 
        data: Object or pd.DataFrame
        func_compute_metric: function name, either attribute of data (if object) to operate on data (if dataframe)
        metrics : dict specifying all the metrics to be plotted and how to compute it (e.g. the trial outcomes to be counted in the numerator/denominator)
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
    
    
    # Empty dict to store results
    results={} 

    legend_flag = True # Needed to suppress repeats of legend entries appearing for traces across subplots

    # Loop through all metrics, compute each metric split by conditions, store and plot results
    iMetric=0

    for metric in metrics:
        iMetric+=1
        
        # Call func_compute_metric of object data with its arguments: e.g. behavior - behavior_metrics.outcome_rate_by_conditions
        if isinstance(data, object) and not isinstance(data, type) and not isinstance(data, pd.DataFrame):
            results[metric] = call_object_func(data, func_compute_metric,
                                                    conditions=conditions, metric_info=metrics[metric])
         
        # If data is a dataframe, call func_compute_metric directly with its arguments: e.g. general.mean_by_condition
        elif isinstance(data, pd.DataFrame):
            results[metric] = func_compute_metric(data, conditions=conditions, metric_info=metrics[metric])
        
        # Store results in 
        results[metric] = results[metric].sort_values(by=x_label)
        
        if(split_label != ''):
            
            # Subselect results only for subset of trials with condition-value specified by split_label-split_cond
            results[metric]=results[metric][results[metric][split_label]==split_cond].reset_index(drop=True)
            
            # Get line-color for the condition-value specified by split_label-split_cond
            try:
                line_color = color_map[split_label][split_cond]
            except:
                line_color='black'
                
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
            Mean=('Value','mean'),
            SE = ('Value',standard_error)
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
        if(column_num_type=='automatic'):
            column_num=iMetric # If column number specified, add trace to that column (used for plotting different subjects). Else add to column iMetric
        if(row_num_type == 'automatic'):
            row_num = 1
        if(make_scatter_plot==False):
            
            fig.add_trace(go.Scatter(x=df_summary[x_label],y=df_summary['Mean'],error_y=dict(array=df_summary['SE']), line=dict(width=4,color=line_color), marker = dict(size=15,color=x_colors), name = 'Mean +/- SE '+str(split_cond), legendgroup = 'Mean +/- SE '+str(split_cond), showlegend = legend_flag), row = row_num,col = column_num)
        
        elif(make_scatter_plot==True):
            
            fig.add_trace(go.Scatter(x=df_summary['Mean'][[df_summary.index[0]]], y=df_summary['Mean'][[df_summary.index[-1]]],error_x=dict(array=[df_summary['SE'][[df_summary.index[0]]]]),error_y=dict(array=[df_summary['SE'][[df_summary.index[-1]]]]), marker = dict(size=20,color=x_colors), name = 'Mean +/- SE '+str(split_cond), legendgroup = 'Mean +/- SE '+str(split_cond), showlegend = legend_flag), row = row_num,col = column_num)
            
#             lower_lim = np.min([df_summary['Mean'][0]- 4*np.std(df_summary['Mean'][0]), df_summary['Mean'][1]- 4*np.std(df_summary['Mean'][1])])
            lower_lim = np.min([df_summary['Mean'][0]- 30, df_summary['Mean'][1]- 30])
            if(lower_lim<0):
                lower_lim=0
        
#             upper_lim = np.max([df_summary['Mean'][0]+ 4*np.std(df_summary['Mean'][0]), df_summary['Mean'][1]+ 4*np.std(df_summary['Mean'][1])])
            
            upper_lim =np.max([df_summary['Mean'][0]+ 30, df_summary['Mean'][1]+ 30])
            if(upper_lim>100):
                upper_lim=100
            fig.add_trace(go.Scatter(x=[lower_lim,upper_lim],y=[lower_lim,upper_lim],mode='lines',line=dict(color='black')), row = row_num,col = column_num)
            
        
        ## Plot individual sessions:

        if(plot_individual_sessions):

            all_sessions = np.sort(results[metric]['Session'].unique())

            session_colors=generate_color_list(num_colors = len(all_sessions), cmap_name = 'turbo', output_format = 'hex') # Gets list of colors
            
            session_num = 0
            for session in all_sessions:
                results_session = results[metric][results[metric]['Session']==session].reset_index(drop=True)
                results_session=results_session.dropna(subset=['Value'])
                results_session[x_label] = pd.Categorical(results_session[x_label], categories=conditions[x_label], ordered=True)
                results_session = results_session.sort_values(by=x_label)
                
                if(make_scatter_plot==False):
                    
                    fig.add_trace(go.Scatter(x = results_session[x_label],y = results_session['Value'], mode='lines', line=dict(width=0.5, dash = dash_type, color=session_colors[session_num]), name = str(session)+' '+str(split_cond), legendgroup = str(session)+' '+str(split_cond), showlegend = legend_flag), row = row_num, col = column_num)
                
                elif(make_scatter_plot==True):
                    
                    fig.add_trace(go.Scatter(x=results_session['Value'][[results_session.index[0]]], y=results_session['Value'][[results_session.index[-1]]], mode='markers', marker=dict(size=10, color=session_colors[session_num]), name = str(session)+' '+str(split_cond), legendgroup = str(session)+' '+str(split_cond), showlegend = legend_flag), row = row_num, col = column_num)
                    
                    
                
                    
                session_num +=1

        legend_flag = False 
    
    # Clean up figure:
    width=300*len(metrics)
    fig.update_layout(width=width,height=400)      
    if(make_scatter_plot):
        fig.update_xaxes(title_text=str(results_session[x_label][results_session.index[0]]))
        fig.update_yaxes(title_text=str(results_session[x_label][results_session.index[-1]]))
    else:
        fig.update_xaxes(title_text=x_label)
    
    return fig


def plot_reach_by_condition(fig, data,   conditions, x_label, metric = 'Velocity Profile', event_to_time_lock = 'Delay',window = [0,1000], fs=100, split_label = '', split_cond = '', plot_individual_sessions = True, column_num=1, row_num=1, subplot_num_type='automatic',color_map='black',dash_map='solid'):
    
    """
    Function to plot velocity profiles and trajectories by condition
    
    """
    
    # Empty dict to store results
    results=[]

    legend_flag = True # Needed to suppress repeats of legend entries appearing for traces across subplots
       
#      # Get velocity profile across trials for each condition:
#     results = get_velocity_profile(data, conditions = conditions, event_to_time_lock = event_to_time_lock, window = window, fs = fs)
    
    # Create combinations of values for each condition
    for value_combination in itertools.product(*conditions.values()):
        condition_dict = dict(zip(conditions.keys(), value_combination)) # Dict with each combination of condition-values
        # Get the mean velocity profile for each condition
        filtered_df = data
        
        for condition, value in condition_dict.items():
            filtered_df = filtered_df[filtered_df[condition] == value].reset_index(drop=True)
            
        if(metric == 'Velocity Profile'):
            vel_profile = np.mean([filtered_df['Velocity Profile'][i][int((filtered_df[event_to_time_lock][i]+window[0])*fs/1000):int((filtered_df[event_to_time_lock][i]+window[1])*fs/1000)] for i in range(len(filtered_df))],0)
            # Append results to the list
            results.append({**condition_dict, 'Value': vel_profile})
            
        elif(metric == 'Trajectory'):
            
            trajectories = ([filtered_df['Trajectory'][i][:,int((filtered_df[event_to_time_lock][i]+window[0])*fs/1000):int((filtered_df[event_to_time_lock][i]+window[1])*fs/1000)] for i in range(len(filtered_df))])
            # Append results to the list
            results.append({**condition_dict, 'Value': trajectories})

    # Create a DataFrame from the results
    results = pd.DataFrame(results)
    
    results = results.sort_values(by=x_label)
    
    
    if(split_label != ''):

        # Subselect results only for subset of trials with condition-value specified by split_label-split_cond
        results=results[results[split_label]==split_cond].reset_index(drop=True)
        
        # Get line-color for the condition-value specified by split_label-split_cond
        try:
            line_color = color_map[split_label][split_cond]
        except:
            line_color='black'

        # dash type not specified for certain conditions
        try:
            dash_type = dash_map[split_label][split_cond]
        except: 
            dash_type = 'solid'

    else:
        line_color = 'black' # default line color for mean+/-SE
        dash_type = 'solid'

    ## Plot Mean +/- SE    

    
    # Add trace
    if(subplot_num_type=='automatic'):
     # If column number specified, add trace to that column (used for plotting different subjects).
     # Else add to column iMetric
        column_num=iMetric
        row_num=1
        
        
    all_unique_labels=np.unique(results[x_label])
    for i in all_unique_labels:
        line_color = color_map[x_label][i]
        results_sub = results[results[x_label]==i].reset_index(drop=True)
        
        if(metric == 'Velocity Profile'):
            x_values=np.linspace(window[0],window[1],len(results_sub['Value'][0]))
            stdshade(fig, results_sub['Value'], x_values=x_values, row=row_num, col=column_num, color=line_color, name=str(i), legend_flag=legend_flag)
        
        elif(metric == 'Trajectory'):
            
            
            traj_sub = [elem for elem in results_sub['Value'][0] if elem.shape == (2,np.diff(window)[0]*fs/1000)]
            traj_sub=np.array(traj_sub)
            x_values = np.mean(traj_sub[:,0,:],0)
            x_error    = np.std(traj_sub[:,0,:],0)/np.sqrt(np.shape(results_sub['Value'])[0])
            y_values = np.mean(traj_sub[:,1,:],0)
            y_error    = np.std(traj_sub[:,1,:],0)/np.sqrt(np.shape(results_sub['Value'])[0])
            
            fig.add_trace(go.Scatter(x=x_values,y=y_values, error_x=dict(type='data', array=x_error, visible=True),
    error_y=dict(type='data', array=y_error, visible=True), line=dict(color=line_color)),row=row_num, col=column_num)
                             

        
    return fig
        

    
    
def plot_PSTH_by_condition(fig, spike_count_train, train_labels, spike_count_test, test_labels, color_map, axis_method='LDA', train_window=[750, 1000], plot_bin_size=1, row_num=1, column_num=1, x_offset=-500, center_projections_flag=True,legend_flag=False, line_width=1,line_dash='solid',smooth_flag=False,subtract_baseline_PSTH=False,baseline_label = 0, cumulative_sum_flag=False, zscore_flag=False, use_axis=None, return_proj=False):
    """
    Plots Peri-Stimulus Time Histogram (PSTH) by condition for test data projected onto a 'Condition Axis' based on the specified axis method and condition labels.
    
    Parameters:
    - fig: The figure object to plot on.
    - spike_count_train: 3D array of spike counts in the training set (trials x neurons x time).
    - train_labels: Labels corresponding to each trial in the training set.
    - spike_count_test: 3D array of spike counts in the test set (trials x neurons x time).
    - test_labels: Labels corresponding to each trial in the test set.
    - color_map: A dictionary or list of colors for each condition.
    - axis_method: The method to calculate the axis for dimensionality reduction (default is 'LDA').
    - train_window: Time window (start, end) to consider for spike counting.
    - plot_bin_size: The size of the bins to use for the plot (default is 1 ms).
    - row_num: The row number for subplots (default is 1).
    - column_num: The column number for subplots (default is 1).
    """
    
    # Bin the training spike counts within the specified time window (e.g., [750, 1000] ms)
    spike_count_bin = np.nanmean(spike_count_train[:,:,train_window[0]:train_window[1]], 2)  # Mean across time window
    
    if(zscore_flag):
        spike_count_bin=zscore(spike_count_bin,axis=0)
    
    if(use_axis is None):
        # Get the condition axis based on the selected axis method (e.g., LDA, PCA, etc.)
        cond_axis = get_condition_axes(spike_count_bin, train_labels, axis_method=axis_method)
    else:
        cond_axis=use_axis
        
    # Bin the test spike counts using the specified bin size
    if(smooth_flag==False):
        spikes_binned = bin_spike_counts(spike_count_test, bin_size=plot_bin_size)
         # Define the x-values for plotting based on the bin size and number of time points
        x_values = (np.arange(0, np.shape(spikes_binned)[2]) * plot_bin_size + (plot_bin_size / 2)) +x_offset
    
    else:
        spikes_binned = bin_spike_counts(spike_count_test, bin_size=1) # Bin size =1 if smoothing 
         # Define the x-values for plotting based on the bin size and number of time points
        x_values = (np.arange(0, np.shape(spikes_binned)[2]) * 1 + (1 / 2)) +x_offset
    
    if(zscore_flag):
        spikes_binned=zscore(spikes_binned,axis=0)
        
    # Project the binned test data onto the condition axis
    proj_data = project_on_axis(np.transpose(spikes_binned, [2, 0, 1]), cond_axis[:, 0]/np.linalg.norm(cond_axis[:, 0]),center_projections=center_projections_flag)
    
    
   
    # Get all unique labels for the test data
    all_test_labels = np.unique(test_labels)
    
    # Loop through each unique label in the test labels
    for i in all_test_labels:
        # For each condition (label), get the corresponding projected data
        proj_data_for_condition = (proj_data[:, test_labels == i].T) * (1 / plot_bin_size)  # Convert to spikes per second
        
        if(subtract_baseline_PSTH): # Used currently for subtracting no-change psth for V4:  
            proj_data_baseline = (proj_data[:, test_labels == baseline_label].T) * (1 / plot_bin_size)
            if(cumulative_sum_flag==True):
                proj_data_for_condition=np.cumsum(proj_data_for_condition-np.nanmean(proj_data_baseline,0),axis=1)
            else:
                proj_data_for_condition=(proj_data_for_condition-np.nanmean(proj_data_baseline,0))

        if(smooth_flag==True):
            proj_data_for_condition = smooth((proj_data[:, test_labels == i].T) *1000,plot_bin_size)  # Smooth + Convert to spikes per second
            if(subtract_baseline_PSTH):
                proj_data_baseline = smooth((proj_data[:, test_labels == baseline_label].T) *1000,plot_bin_size)
                if(cumulative_sum_flag==True):
                    proj_data_for_condition=np.cumsum(proj_data_for_condition-np.nanmean(proj_data_baseline,0),axis=1)
                else:
                    proj_data_for_condition=(proj_data_for_condition-np.nanmean(proj_data_baseline,0))
                
        # Plot the data for this label using the stdshade function (which handles shaded error bars)
        stdshade(fig, proj_data_for_condition, x_values=x_values, row=row_num, col=column_num, 
                 color=color_map[i], name=str(i),legend_flag=legend_flag,line_width=line_width,line_dash=line_dash)
        
    if (return_proj):
        return cond_axis[:,0], proj_data
    else:
        return cond_axis[:, 0]

def make_decoding_figure(fig, df_behav, spikes, all_dates, subject_name, train_condition , test_condition, train_window=None, test_window=[0,1400], bin_size=50, subselect_conditions={}, subsample_flag=True, line_color='black', plot_individual_sessions=False, legend_label='', all_split_conditions = None,x_offset=-500):
    """
    Generate decoding figure based on the provided parameters.

    Parameters:
    - fig : plotly figure 
    - df_behav: DataFrame containing the behavioral data.
    - spikes: Dictionary containing spike data for each subject and date.
    - all_dates: List of all dates for the sessions.
    - subject_name: Name of the subject (e.g., 'Pumbaa').
    - test_window: Time window for testing (e.g., [0, 1400]).
    - train_condition: Condition used to train decoder 
    - test_condition: Condition used to test decoder
    - train_window: Time window for training (None if train/test on each bin, or specify start/end e.g. [750,1000])
    - bin_size: Size of the bin for decoding.
    - subselect_conditions: Conditions for subsetting trials.
    - subsample_flag: Whether or not to subsample the data.
    - line_color: Color of the lines in the plot.
    - plot_individual_sessions: Whether to plot individual session results.
    - legend_label : Label to be used in the legend 
    """
    accuracies=[]
    color_scale_sess=hue_rotate_scale(line_color,len(all_dates))
    i_sess=-1
    for date in all_dates:
        i_sess=i_sess+1
        df_behav_date = df_behav[df_behav['Session Name']==date].reset_index(drop=True)
        neural_data = spikes[subject_name][date]
        neural_data = neural_data[:,:,test_window[0]:test_window[1]]
        accuracies.append(decode_by_sess(df_behav_date, neural_data, subselect_conditions, train_condition, test_condition, bin_size, train_window, subsample_flag, all_split_conditions))
        
        if(plot_individual_sessions):
            fig.add_trace(go.Scatter(showlegend=True, line=dict(width=2),mode='lines',marker_color=line_color,x=x_offset+np.arange(len(accuracies[-1]))*bin_size + (bin_size/2), y=accuracies[-1]))
    
    fig.add_trace(go.Scatter(name=legend_label,legendgroup = legend_label, line=dict(width=4), marker_color=line_color,x=x_offset+test_window[0]+np.arange(len(accuracies[0]))*bin_size + (bin_size/2), y=np.nanmean(accuracies,0), error_y=dict(array = np.std(accuracies,0)/np.sqrt(len(all_dates)))))
    
    fig['layout']['title'] = subject_name 
    fig['layout']['yaxis']['title'] = 'Decoding accuracy'
    fig['layout']['xaxis']['title'] = 'Time from Target Onset (ms)' 
    
import plotly.colors as pc
import colorsys

def make_darker_scale(base_color, n=5):
    r, g, b = mcolors.to_rgb(base_color) 
    colors = [mcolors.to_hex((r*(1 - i/n), g*(1 - i/n), b*(1 - i/n)))
              for i in range(n)]
    return colors

def hue_rotate_scale(base_color, n=5, total_degrees=60):
    """Generate a scale by rotating hue around the color wheel."""
    # convert base color to HLS (Hue, Lightness, Saturation)
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    colors = []
    for i in range(n):
        # rotate hue by fraction of total_degrees
        new_h = (h - (total_degrees / 360) * (i / (n ))) % 1.0
        new_r, new_g, new_b = colorsys.hls_to_rgb(new_h, l, s)
        colors.append(mcolors.to_hex((new_r, new_g, new_b)))

    return colors
def make_decoding_figure_single_bin(fig, df_behav, spikes, all_dates, subject_name, train_condition , test_condition, train_window=None, test_window=[500,1000],bin_size=500, subselect_conditions={}, subsample_flag=True, line_color='black', plot_individual_sessions=False, legend_label='', all_split_conditions = None,x_label=['Target']):
    """
    Generate decoding figure based on the provided parameters.

    Parameters:
    - fig : plotly figure 
    - df_behav: DataFrame containing the behavioral data.
    - spikes: Dictionary containing spike data for each subject and date.
    - all_dates: List of all dates for the sessions.
    - subject_name: Name of the subject (e.g., 'Pumbaa').
    - test_window: Time window for testing (e.g., [0, 1400]).
    - train_condition: Condition used to train decoder 
    - test_condition: Condition used to test decoder
    - train_window: Time window for training (None if train/test on each bin, or specify start/end e.g. [750,1000])
    - subselect_conditions: Conditions for subsetting trials.
    - subsample_flag: Whether or not to subsample the data.
    - line_color: Color of the lines in the plot.
    - plot_individual_sessions: Whether to plot individual session results.
    - legend_label : Label to be used in the legend 
    """
    color_scale_sess=make_darker_scale(line_color,len(all_dates))
    accuracies=[]
    i_sess=-1
    for date in all_dates:
        i_sess=i_sess+1
        df_behav_date = df_behav[df_behav['Session Name']==date].reset_index(drop=True)
        neural_data = spikes[subject_name][date]
        neural_data = neural_data[:,:,test_window[0]:test_window[1]]
        accuracies_session=(decode_by_sess(df_behav_date, neural_data, subselect_conditions, train_condition, test_condition, bin_size, train_window, subsample_flag, all_split_conditions))
        accuracies.append([np.max(accuracies_session)])
        
        if(plot_individual_sessions):
            fig.add_trace(go.Scatter(showlegend=True,mode='markers',marker_color=color_scale_sess[i_sess],x=x_label, y=accuracies[-1]))
    
    fig.add_trace(go.Bar(name=legend_label,legendgroup = legend_label,marker=dict(opacity=0.5), marker_color=line_color,x=x_label, y=np.nanmean(accuracies,0), error_y=dict(array = np.std(accuracies,0)/np.sqrt(len(all_dates)))))
    
    fig['layout']['title'] = subject_name 
    fig['layout']['yaxis']['title'] = 'Decoding accuracy'
    
        
def make_ring_plots(fig, df_behav, spikes, all_dates, subject_name, train_condition , test_condition, train_window=[700,1000], test_window=[700,1000], subselect_conditions_train={}, subselect_conditions_test={}, line_color='black', plot_individual_sessions=False, legend_label='',row_num=1,col_num=1,marker_colors=['orange', 'green', 'blue', 'magenta'],line_width=4,line_dash='solid', use_axis=False,special_axis=None, return_space=False, remove_bad_units_flag=True,axis_method="PCA"):
    """
    Generate figure with ring plots (projection onto top 2 dimensions explaining related variance related to a circular variable) based on the provided parameters.

    Parameters:
    - fig : plotly figure 
    - df_behav: DataFrame containing the behavioral data.
    - spikes: Dictionary containing spike data for each subject and date.
    - all_dates: List of all dates for the sessions.
    - subject_name: Name of the subject (e.g., 'Pumbaa').
    - test_window: Time window for testing (e.g., [0, 1400]).
    - train_condition: Condition used to train decoder 
    - test_condition: Condition used to test decoder
    - train_window: Time window for training (None if train/test on each bin, or specify start/end e.g. [750,1000])
    - subselect_conditions: Conditions for subsetting trials.
    - line_color: Color of the lines in the plot.
    - plot_individual_sessions: Whether to plot individual session results.
    - legend_label : Label to be used in the legend 
    """
    projections=[]
    target_space=[]
    for date in all_dates:
        
        df_behav_date = df_behav[df_behav['Session Name']==date].reset_index(drop=True)
        neural_data = spikes[subject_name][date]
        if(remove_bad_units_flag):
            neural_data_good_units = remove_bad_units(neural_data,1)
        else:
            neural_data_good_units=neural_data
            
#         neural_data = neural_data[:,:,test_window[0]:test_window[1]]
        if(return_space):
            proj, target_space_date =ring_plot_by_sess(df_behav_date, neural_data_good_units, subselect_conditions_train,subselect_conditions_test , train_condition, test_condition, train_window, test_window,use_axis=use_axis, special_axis=special_axis, return_space=return_space,axis_method=axis_method)
            
            projections.append(proj)
            target_space.append(target_space_date)
        
        else:
            
            proj = ring_plot_by_sess(df_behav_date, neural_data_good_units, subselect_conditions_train,subselect_conditions_test , train_condition, test_condition, train_window, test_window,use_axis=use_axis, special_axis=special_axis, return_space=return_space,axis_method=axis_method)
            projections.append(proj)
        
        if(plot_individual_sessions):
            fig.add_trace(go.Scatter(showlegend=True, line=dict(width=.5),marker_color=line_color,x=projections[-1][0], y=projections[-1][1]), row=row_num, col=col_num)
    
    projections=np.array(projections)
    
    fig.add_trace(go.Scatter(name=legend_label,legendgroup = legend_label, line=dict(color=line_color,width=line_width,dash=line_dash) ,marker=dict(size=15,color=marker_colors), x=np.nanmean(projections[:,:,0],0), y=np.nanmean(projections[:,:,1],0), error_y=dict(array = np.std(projections[:,1],0)/np.sqrt(len(all_dates))), error_x=dict(array = np.std(projections[:,0],0)/np.sqrt(len(all_dates)))), row=row_num, col=col_num)
    
    
    fig['layout']['title'] = subject_name 
    fig['layout']['yaxis']['title'] = 'Dim. 2'
    fig['layout']['xaxis']['title'] = 'Dim. 1' 

    if(return_space):
        return target_space

def plot_single_neuron_direction_tuning(fig, spike_count, direction_labels, color_map, window=[750, 1000], row_num=1, column_num=1, legend_flag=False):
    
    # Bin the spike counts within the specified time window (e.g., [750, 1000] ms)
    spike_count_bin = np.nanmean(spike_count[:,:,window[0]:window[1]], 2)*1000  # Mean across time window
    R = []
    theta = []
    num_neurons = np.shape(spike_count_bin)[1]
    # Get the mean spike count per direction (across trials)
    mean_spike_count_per_direction= compute_mean_spike_count(spike_count_bin, direction_labels)
    p_values_tuning = test_spike_counts_anova(spike_count_bin, direction_labels)
    
    col_sig = ['lightgrey','black']
    for i_neuron in range(num_neurons):
        # Get the average FR for each direction for each neuron: 
        all_directions = np.unique(direction_labels)
        
        # Convert vectors to cartesion form and add: 
        V_x = 0
        V_y = 0
        for i_direction in range(len(all_directions)):
            V_x = V_x + mean_spike_count_per_direction[i_direction,i_neuron]*np_cosd(all_directions[i_direction])
            V_y = V_y + mean_spike_count_per_direction[i_direction,i_neuron]*np_sind(all_directions[i_direction])
        
        R.append(np.sqrt(V_x**2 + V_y**2))
        theta.append(np.degrees(np.arctan2(V_y,V_x)))
        p_val_i=p_values_tuning[i_neuron]
        fig.add_trace(go.Scatter(x=[V_x], y=[V_y],marker_color=col_sig[p_val_i<0.05], mode='markers',showlegend=legend_flag),row=row_num,col=column_num)
    
    theta = convert_to_0_to_360(theta)
    return fig, R, theta, p_values_tuning



# RT vs Neural change time correlation: 


from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

def plot_rt_vs_signal(RT, signal, title_prefix="RT vs signal"):
    """
    Plot RT vs neural/behavioral signal time, compute Spearman correlation and R^2.

    Parameters
    ----------
    RT : np.ndarray
        Reaction times per trial
    signal : np.ndarray
        Signal times or strength per trial (same length as RT) 
    title_prefix : str
        Optional prefix for plot title

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Scatter plot figure
    rho : float
        Spearman rank correlation coefficient
    r2_percent : float
        Percentage of variance in RT explained by signal time
    """
    # Remove NaNs
    mask = ~np.isnan(RT) & ~np.isnan(signal)
    rt_clean = RT[mask]
    sig_clean = signal[mask]

    # Spearman correlation
    rho, _ = spearmanr(sig_clean, rt_clean)

    # Linear regression R^2
    X = sig_clean.reshape(-1, 1)
    y = rt_clean
    model = LinearRegression().fit(X, y)
    r2_percent = model.score(X, y) * 100

    # Scatter plot
    fig = go.Figure(go.Scatter(
        x=sig_clean,
        y=rt_clean,
        mode='markers',
        marker=dict(color='blue', size=6),
        name='Trials'
    ))

    fig.update_layout(
        xaxis_title='Neural signal',
        yaxis_title='Reaction time (ms)',
        title=f'{title_prefix} (Spearman rho = {rho:.2f}, R² = {r2_percent:.1f}%)',
        template='simple_white'
    )

    fig.show()

    return fig, rho, r2_percent
