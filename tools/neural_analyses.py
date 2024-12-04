'''
Module to perform neural analyses

'''

## Import stuff
import itertools
import pandas as pd
import numpy as np
from scipy.linalg import svd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler


def get_condition_axes(spike_count, labels, axis_method='PCA'):
    '''
    Wrapper function to compute the 'condition' axes for dimensionality reduction or classification.
    It computes the axes based on the specified method, which can be either 'PCA' or 'LDA'.

    Parameters:
    - spike_count (ndarray): A matrix of spike counts (num_trials x num_neurons).
    - labels (ndarray): An array of labels corresponding to the conditions (num_trials).
    - axis_method (str, optional): The method to use for computing the axes. Options:
        - 'PCA' (default): Principal Component Analysis (PCA) for dimensionality reduction.
        - 'LDA': Linear Discriminant Analysis (LDA) for maximizing class separability.
    
    Returns:
    - condition_axes (ndarray): The computed axes (dimensionality reduction or discriminant axes).
    '''
    
    # Step 1: Get the number of unique labels (conditions) - 1
    N = len(np.unique(labels))-1  
    
    # Step 2: Choose the axis computation method based on 'axis_method'
    
    if axis_method == 'PCA':
        # If PCA is chosen for dimensionality reduction:
        
        # Compute the mean spike count for each condition (class)
        mean_spike_count = compute_mean_spike_count(spike_count, labels)
        
        # Compute the top N principal components (axes) using PCA
        condition_axes = compute_PCA_axes(mean_spike_count, N)
    
    elif axis_method == 'LDA':
        # If LDA is chosen for maximizing class separability:
        
        # Compute the top N Linear Discriminant Axes using LDA
        condition_axes = compute_LDA_axes(spike_count, labels, N=N)
    
    # Step 3: Return the computed axes (PCA or LDA)
    return condition_axes


def compute_mean_spike_count(spike_count, labels):
    """
    Computes the mean spike count for each condition and channel.
    
    Parameters:
    - spike_count: 2D NumPy array of shape (num_trials x num_channels), where each element represents the spike count for a given trial and channel.
    - labels: 1D NumPy array of shape (num_trials,), where each element is the label corresponding to the condition of each trial.

    Returns:
    - mean_spike_count: 2D NumPy array of shape (num_conditions x num_channels), where each element represents the mean spike count for a given condition and channel.
    """
    
    # Get the unique conditions (labels)
    unique_labels = np.unique(labels)
    
    # Initialize an array to hold the mean spike counts for each condition and channel
    mean_spike_count = np.zeros((len(unique_labels), spike_count.shape[1]))
    
    # Iterate over each condition (label)
    for i, label in enumerate(unique_labels):
        # Get the indices of trials that belong to the current condition
        condition_trials = np.where(labels == label)[0]
        
        # Compute the mean spike count for each channel in the current condition
        mean_spike_count[i, :] = np.mean(spike_count[condition_trials, :], axis=0)
    
    return mean_spike_count


def compute_PCA_axes(X, N=1):
    """
    Computes the top N principal components (PCA) of a dataset using SVD.

    Parameters:
    - X: 2D NumPy array of shape (num_observations x num_variables),
         where each row is an observation and each column is a variable.
    - N: The number of top principal components to return.

    Returns:
    - top_n_components: 2D NumPy array of shape (num_variables x N),
                         where each column is a principal component.
    """
    
    # Step 1: Standardize the data (zero mean, unit variance)
    X_centered = X - np.mean(X, axis=0)  # Subtract the mean of each column
    X_standardized = X_centered / np.std(X, axis=0)  # Divide by the std of each column
    
    # Step 2: Perform Singular Value Decomposition (SVD)
    # SVD decomposes X = U * S * V^T
    U, S, Vt = svd(X_standardized, full_matrices=False)  # full_matrices=False for economy size
    
    # Step 3: Select the top N principal components (corresponding to the top N columns of Vt)
    top_n_components = Vt[:N, :].T  # Take the first N rows of Vt and transpose to get principal components
    
    return top_n_components



def compute_LDA_axes(X, labels, N=1):
    """
    Function to compute the top N LDA axes that best separate the classes.
    
    Parameters:
    - X(ndarray): A (num_observations x num_variables) array 
    - labels (ndarray): A (num_observations,) array of class labels.
    - N (int): Number of top linear combinations to return.
    
    Returns:
    - LDA_axes (ndarray): The top N linear combinations of variables (the directions that
                                       best separate the classes).
    """
    # Step 1: Standardize the data (zero mean, unit variance)
    X_centered = X - np.mean(X, axis=0)  # Subtract the mean of each column
    X_standardized = X_centered / np.std(X, axis=0)  # Divide by the std of each column

    # Perform Linear Discriminant Analysis (LDA)
    lda = LDA(n_components=N)
    lda.fit(X_standardized, labels)

    # The coefficients (weights) of the linear discriminants
    # These coefficients define the linear combinations of neurons
    # that best separate the classes.
    LDA_axes = lda.coef_

    return LDA_axes.T


def project_on_axis(X, axes, data_for_standardizing='none', X_orig=0, center_projections = True):
    """
    Projects the input data `X` onto the given axes after standardizing the data.
    
    Parameters:
    - X (ndarray): The data to be projected (num_trials x num_features) or (num_time_points x num_trials x num_features)
    - axes (ndarray): The axes onto which to project the data (num_axes x num_features).
    - data_for_standardizing (str, optional): Indicates which data to use for standardization. 
                                              Options are:
                                              - 'current': Standardize `X` using its own mean and std.
                                              - 'original': Standardize `X` using the mean and std of `X_orig`.
                                              - 'none' (default) : Do not standardize 'X'
    - X_orig (ndarray, optional): The original dataset used for standardizing when `data_for_standardizing='original'`.
    - center_projections (Boolean): Subtract mean of projected_data 
    
    Returns:
    - projected_data (ndarray): The data projected onto the given axes 
                                (num_trials x num_axes) or (num_time_points x num_trials x num_features).
    """
    
    # Step 1: Standardize the data based on the chosen standardization method
    if data_for_standardizing == 'current':
        # If using the current dataset X for standardization:
        X_centered = X - np.nanmean(X, axis=0)  # Subtract the mean of each feature (column)
        X_standardized = X_centered / np.nanstd(X, axis=0)  # Divide by the std of each feature (column)
    
    elif data_for_standardizing == 'original':
        # If using the original dataset X_orig for standardization:
        X_centered = X - np.nanmean(X_orig, axis=0)  # Subtract the mean of each feature (column) of X_orig
        X_standardized = X_centered / np.nanstd(X_orig, axis=0)  # Divide by the std of each feature (column) of X_orig
    
    elif data_for_standardizing == 'none':
        X_standardized = X  # No standardization
    
    # Step 2: Project the standardized data onto the provided axes
    projected_data = np.dot(X_standardized, axes.T)  # Perform matrix multiplication to project onto axes
    
    if center_projections:
        projected_data = projected_data - np.mean(projected_data)
        
    return projected_data


def bin_spike_counts(X, bin_size):
    """
    Bins the spike count matrix X along the time dimension with the given bin size.
    
    Parameters:
    - X (ndarray): The spike count matrix with shape (num_trials, num_channels, num_time_points).
    - bin_size (int): The size of each bin (number of time points per bin).
    
    Returns:
    - binned_X (ndarray): The binned spike count matrix with shape (num_trials, num_channels, num_bins).
    """
    # Get the original shape of the matrix
    num_trials, num_channels, num_time_points = X.shape
    
    # Calculate the number of bins (time points divided by bin size)
    num_bins = num_time_points // bin_size
    
    # Reshape X so that time dimension is split into bins
    # This will reshape X into (num_trials, num_channels, num_bins, bin_size)
    reshaped_X = X.reshape(num_trials, num_channels, num_bins, bin_size)
    
    # Sum the values in each bin along the 3rd axis (axis=3)
    binned_X = np.sum(reshaped_X, axis=3)
    
    return binned_X

    
    


