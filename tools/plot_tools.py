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


  
def stdshade(fig, data, x_values, error_type='sem', row=1, col=1, color='#FF5733', name="Mean with Shaded Error", xlabel="Time", ylabel="",legend_flag=False):
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
    mean_signal = np.mean(data_array, axis=0)
    
    # Calculate the error (std or sem)
    if error_type == 'std':
        error = np.std(data_array, axis=0)
    elif error_type == 'sem':
        error = np.std(data_array, axis=0) / np.sqrt(len(data))
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
        line=dict(color=color, width=1),
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
        showlegend=False
    ), row=row, col=col)
    
    # Update layout for the subplot
    fig.update_xaxes(title_text=xlabel, row=row, col=col)
    fig.update_yaxes(title_text=ylabel, row=row, col=col)


    

"""
Define figure plotting functions

"""

def plot_metrics_by_condition(fig, data, func_compute_metric, metrics, conditions, x_label, split_label = '', split_cond = '', plot_individual_sessions = True, column_num=1, column_num_type='automatic',color_map='black',dash_map='solid',make_scatter_plot=False):

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
        if(make_scatter_plot==False):
            
            fig.add_trace(go.Scatter(x=df_summary[x_label],y=df_summary['Mean'],error_y=dict(array=df_summary['SE']), line=dict(width=4,color=line_color), marker = dict(size=15,color=x_colors), name = 'Mean +/- SE '+str(split_cond), legendgroup = 'Mean +/- SE '+str(split_cond), showlegend = legend_flag), row = 1,col = column_num)
        
        elif(make_scatter_plot==True):
            
            fig.add_trace(go.Scatter(x=df_summary['Mean'][[df_summary.index[0]]], y=df_summary['Mean'][[df_summary.index[-1]]],error_x=dict(array=[df_summary['SE'][0]]),error_y=dict(array=[df_summary['SE'][1]]), marker = dict(size=15,color=x_colors), name = 'Mean +/- SE '+str(split_cond), legendgroup = 'Mean +/- SE '+str(split_cond), showlegend = legend_flag), row = 1,col = column_num)
            
#             lower_lim = np.min([df_summary['Mean'][0]- 4*np.std(df_summary['Mean'][0]), df_summary['Mean'][1]- 4*np.std(df_summary['Mean'][1])])
            lower_lim = np.min([df_summary['Mean'][0]- 30, df_summary['Mean'][1]- 30])
            if(lower_lim<0):
                lower_lim=0
        
#             upper_lim = np.max([df_summary['Mean'][0]+ 4*np.std(df_summary['Mean'][0]), df_summary['Mean'][1]+ 4*np.std(df_summary['Mean'][1])])
            
            upper_lim =np.max([df_summary['Mean'][0]+ 30, df_summary['Mean'][1]+ 30])
            if(upper_lim>100):
                upper_lim=100
            fig.add_trace(go.Scatter(x=[lower_lim,upper_lim],y=[lower_lim,upper_lim],mode='lines',line=dict(color='black')), row = 1,col = column_num)
            
        
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
                    
                    fig.add_trace(go.Scatter(x = results_session[x_label],y = results_session['Value'], mode='lines', line=dict(width=0.5, dash = dash_type, color=session_colors[session_num]), name = str(session)+' '+str(split_cond), legendgroup = str(session)+' '+str(split_cond), showlegend = legend_flag), row = 1, col = column_num)
                
                elif(make_scatter_plot==True):
                    
                    fig.add_trace(go.Scatter(x=results_session['Value'][[results_session.index[0]]], y=results_session['Value'][[results_session.index[-1]]], mode='markers', marker=dict(color=session_colors[session_num]), name = str(session)+' '+str(split_cond), legendgroup = str(session)+' '+str(split_cond), showlegend = legend_flag), row = 1, col = column_num)
                    
                    
                
                    
                session_num +=1

        legend_flag = False 
    
    # Clean up figure:
    width=300*len(metrics)
    fig.update_layout(width=width,height=400)      
    if(make_scatter_plot):
        fig.update_xaxes(title_text=results_session[x_label][results_session.index[0]])
        fig.update_yaxes(title_text=results_session[x_label][results_session.index[-1]])
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
        

    
    
def plot_PSTH_by_condition(fig, spike_count_train, train_labels, spike_count_test, test_labels, color_map, axis_method='LDA', train_window=[750, 1000], plot_bin_size=1, row_num=1, column_num=1, x_offset=-500):
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
    
    # Get the condition axis based on the selected axis method (e.g., LDA, PCA, etc.)
    cond_axis = get_condition_axes(spike_count_bin, train_labels, axis_method=axis_method)
    
    # Bin the test spike counts using the specified bin size
    spikes_binned = bin_spike_counts(spike_count_test, bin_size=plot_bin_size)
    
    # Project the binned test data onto the condition axis
    proj_data = project_on_axis(np.transpose(spikes_binned, [2, 0, 1]), cond_axis[:, 0])
    
    # Define the x-values for plotting based on the bin size and number of time points
    x_values = (np.arange(0, np.shape(spikes_binned)[2]) * plot_bin_size + (plot_bin_size / 2)) +x_offset
    
    # Get all unique labels for the test data
    all_test_labels = np.unique(test_labels)
    
    # Loop through each unique label in the test labels
    for i in all_test_labels:
        # For each condition (label), get the corresponding projected data
        proj_data_for_condition = (proj_data[:, test_labels == i].T) * (1000 / plot_bin_size)  # Convert to spikes per second
        
        # Plot the data for this label using the stdshade function (which handles shaded error bars)
        stdshade(fig, proj_data_for_condition, x_values=x_values, row=row_num, col=column_num, 
                 color=color_map[i], name=str(i))
