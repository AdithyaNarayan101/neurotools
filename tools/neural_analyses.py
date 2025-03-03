'''
Module to perform neural analyses

'''

## Import stuff
import itertools
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
from tools.general import *
from scipy.linalg import svd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold


def remove_bad_units(spike_count, cutoff):
    
    FR = np.mean(np.sum(spike_count,2)/np.shape(spike_count)[2],0)*1000
    
    good_FR_units = FR>cutoff
    
    return spike_count[:,good_FR_units,:]
    
    
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
   
    elif axis_method == 'delta_mean':
        # If delta_mean is chosen for dimensionality reduction:
        
        # Compute the mean spike count for each condition (class)
        mean_spike_count = compute_mean_spike_count(spike_count, labels)
        
        # Compute the axis connecting the mean of the first and last labels
        condition_axes=np.zeros((np.shape(mean_spike_count)[1],1))
        condition_axes[:,0] = np.transpose(mean_spike_count[-1,:] - mean_spike_count[0,:])
        
    elif axis_method == 'all_ones':
        
        condition_axes=np.ones((np.shape(spike_count)[1],1))
        
    # Step 3: Return the computed axes (PCA or LDA)
    return condition_axes


def test_spike_counts_anova(spike_count, labels):
    num_trials, num_channels = spike_count.shape
    p_values = []

    # Perform ANOVA for each channel
    for channel in range(num_channels):
        # Extract the spike counts for the current channel
        spike_counts_for_channel = spike_count[:, channel]

        # Group the spike counts by the labels (conditions)
        grouped_data = [spike_counts_for_channel[labels == label] for label in np.unique(labels)]
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*grouped_data)
        
        # Append the p-value for this channel
        p_values.append(p_value)

    return np.array(p_values)



def compute_mean_spike_count(spike_count, labels):
    """
    Computes the mean spike count for each condition and channel.
    
    Parameters:
    - spike_count: 2D NumPy array of shape (num_trials x num_channels), where each element represents the spike count for a given trial and channel.
    - labels: 1D NumPy array of shape (num_trials,), where each element is the label corresponding to the condition of each trial.
    - return_stats: If True, perform 

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
    binned_X = np.nansum(reshaped_X, axis=3)
    
    return binned_X

    

def behavior_decoder(neural_data, behavior_labels, train_window=None, bin_size=20, classifier='lda', train_labels=None, cv_folds=2, subsample_flag = False):
    """
    Decode behavior labels from neural data using either a Naive Bayes or LDA classifier with manual cross-validation.
    If `train_window` is provided, the classifier is trained on this window, and tested on each time bin.
    If `train_window` is not provided, the classifier is trained on each time bin and tested on the same bin.

    Parameters:
    neural_data (numpy.ndarray): Neural data of shape (num_trials, num_neurons, num_timepoints).
    behavior_labels (list or numpy.ndarray): Behavior labels of length num_trials (e.g., angles), used for testing.
    train_window (tuple, optional): A tuple (start, end) specifying the time window (in number of timepoints) for training.
    bin_size (int): Size of the time bin to decode the behavior in (in number of timepoints).
    classifier (str): Choose between 'naive_bayes' or 'lda'. Defaults to 'lda'.
    train_labels (list or numpy.ndarray, optional): Labels to use for training the classifier. If None, behavior_labels are used for training.
    cv_folds (int): Number of folds for cross-validation. Defaults to 5.
    subsample_flag(boolean): Subsample to ensure equal number of items in each class before training/testing decoder
    
    Returns:
    List of mean accuracies for each time bin across cross-validation folds.
    """
    
    # Get the number of trials, neurons, and timepoints
    num_trials, num_neurons, num_timepoints = neural_data.shape
    
    # If train_labels is not provided, use behavior_labels for training
    if train_labels is None:
        train_labels = behavior_labels
    
    # Check if the number of behavior labels matches the number of trials
    if len(behavior_labels) != num_trials:
        raise ValueError("The number of behavior labels must match the number of trials in the neural data.")
    if len(train_labels) != num_trials:
        raise ValueError("The number of train_labels must match the number of trials in the neural data.")
    
    # Normalize the neural data (standardization) for all timepoints (needed if we train separately on each bin)
    scaler = StandardScaler()

    # Initialize list to store accuracies for each time bin
    decoding_accuracies = []
    
    # Create bins for testing
    num_bins = num_timepoints // bin_size  # number of bins of size `bin_size`

    # Loop over each bin
    for bin_idx in range(num_bins):
        # Define the start and end index for this bin
        start_idx = bin_idx * bin_size
        end_idx = (bin_idx + 1) * bin_size
        
        # If a train_window is specified, train on that window
        if train_window:
            train_start, train_end = train_window
            if train_start < 0 or train_end >= num_timepoints or train_start >= train_end:
                raise ValueError("The specified train_window exceeds the range of timepoints in the data.")
            # Extract the training data within the train_window
            train_data = neural_data[:, :, train_start:train_end+1]  # Shape: (num_trials, num_neurons, window_size)
            train_data_reshaped = train_data.mean(axis=2)  # Average across time within the window (Shape: (num_trials, num_neurons))
        else:
            # If train_window is not specified, train and test separately on the current bin
            train_data = neural_data[:, :, start_idx:end_idx]  # Data for current time bin
            train_data_reshaped = train_data.mean(axis=2)  # Average across time within the bin (Shape: (num_trials, num_neurons))
        
        # Normalize the training data
        train_data_scaled = scaler.fit_transform(train_data_reshaped)
        
        # Extract the population activity in the current time bin (for testing)
        test_data = neural_data[:, :, start_idx:end_idx].mean(axis=2)  # Shape: (num_trials, num_neurons)
        # Normalize the test data (using the same scaler as for training)
        test_data_scaled = scaler.transform(test_data)
        # Choose the classifier based on the `classifier` argument
        if classifier == 'naive_bayes':
            clf = GaussianNB()
        elif classifier == 'lda':
            clf = LDA()
        else:
            raise ValueError("Invalid classifier specified. Choose 'naive_bayes' or 'lda'.")
        
        # Initialize the KFold cross-validator
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=420)
        
        # List to store the accuracy for each fold
        fold_accuracies = []
        
        # Perform manual cross-validation
        for train_idx, test_idx in kf.split(train_data_scaled):
            # Split the data into train and test folds
            X_train, X_test = train_data_scaled[train_idx], test_data_scaled[test_idx]
            y_train, y_test = train_labels[train_idx], behavior_labels[test_idx]
            
            if(subsample_flag):
                X_train,y_train = subsample_balance_classes(X_train,y_train)
            # Train the classifier on the training fold
            clf.fit(X_train, y_train)
            
            if(subsample_flag):
                X_test,y_test = subsample_balance_classes(X_test,y_test)
                
            # Test the classifier on the test fold
            test_accuracy = clf.score(X_test, y_test)
            fold_accuracies.append(test_accuracy)
            
        # Calculate the mean accuracy across folds for this bin
        mean_fold_accuracy = np.mean(fold_accuracies)
        
        # Store the mean accuracy for this bin
        decoding_accuracies.append(mean_fold_accuracy)
    
    return decoding_accuracies

def decode_by_sess(df_behav_date, neural_data, subselect_conditions, train_condition, test_condition, bin_size, train_window=None, subsample_flag=True, all_split_conditions=None):
    
    """
    Decode the session data based on behavioral and neural data.

    Parameters:
    - df_behav_date: DataFrame containing the behavior data for a particular session.
    - neural_data: Neural data for a particular session.
    - subselect_conditions: Conditions for subsetting trials.
    - train_condition: Condition used to train decoder 
    - test_condition: Condition used to test decoder
    - bin_size: Size of the bin for decoding.
    - train_window: The training window for decoding.
    - subsample_flag: Whether or not to subsample the data.
    - all_split_conditions: Dict with all the sets of conditions being compared. 
                            If not None, then the number of trials is matched across split conditions  
                            
    Returns:
    - acc_sess: Accuracy for the current session.
    """
    
    # Get min number of trials for each split condition
    if(all_split_conditions):
        num_trials_split=[]
        for split_condition in all_split_conditions.keys():
            _, idx_subselect = subselect_trials_idx(df_behav_date,all_split_conditions[split_condition],return_indices=True)
            num_trials_split.append(len(idx_subselect))
        min_trials = np.min(num_trials_split)
    
    # Subselect trials to use for decoding
    if(subselect_conditions):
        df_behav_date, idx_subselect = subselect_trials_idx(df_behav_date,subselect_conditions,return_indices=True)
        neural_data = neural_data[idx_subselect]
    
    # Match number of trials across splits if 'all_split_conditions' was passed
    if(all_split_conditions):
        df_behav_date, idx_match = subsample_trials_df(df_behav_date,min_trials,return_indices=True)
        neural_data = neural_data[idx_match]
#         print(min_trials)
        
    acc_sess = behavior_decoder(neural_data, df_behav_date[test_condition], bin_size=bin_size, train_window=train_window, train_labels = df_behav_date[train_condition], subsample_flag = subsample_flag)
    
    return acc_sess

def ring_plot_by_sess(df_behav_date, neural_data, subselect_conditions_train, subselect_conditions_test, train_condition, test_condition, train_window=[700,1000],test_window=[700,1000]):
    
    """
    Compute projections onto top 2 dimensions for the session data based on behavioral and neural data.

    Parameters:
    - df_behav_date: DataFrame containing the behavior data for a particular session.
    - neural_data: Neural data for a particular session.
    - subselect_conditions_train/test: Conditions for subsetting trials.
    - train_condition: Condition used to find space
    - test_condition: Condition used to project
    - train_window: The training window for finding the subspace.
    - test_window: The testing window for projecting data.
                            
    Returns:
    - projections: Accuracy for the current session.
    """
    
    # Subselect trials to use for finding the axes
    df_behav_date_train, idx_subselect = subselect_trials_idx(df_behav_date,subselect_conditions_train,return_indices=True)
    train_data = neural_data[idx_subselect]
    
    # Subselect trials to use for projecting onto the axes
    df_behav_date_test, idx_subselect = subselect_trials_idx(df_behav_date,subselect_conditions_test,return_indices=True)
    test_data = neural_data[idx_subselect]
    
    # Bin the training spike counts within the specified time window (e.g., [750, 1000] ms)
    spike_count_bin = np.nanmean(train_data[:,:,train_window[0]:train_window[1]], 2)  # Mean across time window
    mean_spike_count_bin = np.mean(spike_count_bin,0)
    train_labels=df_behav_date_train[train_condition]
    
    # Get the condition axis based on the selected axis method (e.g., LDA, PCA, etc.)
    cond_axis = get_condition_axes(spike_count_bin, train_labels, axis_method='PCA')
#     cond_axis=cond_axis/np.linalg.norm(cond_axis)
    
    # Project the test data onto the condition axis
    spike_count_test_bin = np.nanmean(test_data[:,:,test_window[0]:test_window[1]], 2)  # Mean across time window
    test_labels=df_behav_date_test[test_condition]
    
    proj_data = project_on_axis((spike_count_test_bin)-mean_spike_count_bin, np.transpose(cond_axis[:, :2]))
    
    # Get all unique labels for the test data
    all_test_labels = np.unique(test_labels)
    
    # Loop through each unique label in the test labels
    proj_data_for_condition=[]
    for i in all_test_labels:
        # For each condition (label), get the corresponding projected data
        proj_data_for_condition.append(np.nanmean(proj_data[test_labels == i,:],0) * (1000 ))  # Convert to spikes per second
        
    proj_data_for_condition.append(proj_data_for_condition[0])
    

    return proj_data_for_condition



