
'''
Preprocessing script for neural data. Performs binning of spikes for Smith lab data

'''
import numpy as np

def bin_spikes(dat, bin_size=.001, channels=list(range(64)), filter_spike_codes=[1], 
               align_type='start', start_code=70, start_offset=-.5, end_code=11,
               end_offset=1, total_length=2.5, cutoff=False, fs=30000):
    '''
    Bin spikes (works for smith lab data)

    Parameters: 
        dat : dict
            Contains 'spiketimesdiff', 'spikeinfo', 'firstspike', 'trialcodes', etc.
        bin_size : float
            Size of bin in ms (default = 1ms)
        channels : list
            List of channels to include (default 0:63)
        filter_spike_codes : list
            List of sorting codes to include (default [1]) : 0/255 is noise, 1 (or above) is spike
        align_type : str
            'start' or 'end'. Specifies how to align bins (default = 'start')
        start_code : int
            Code for trial start event (default = 70)
        start_offset : int
            Offset in ms from start_code to use for binning (default = -500)
        end_code : int
            Code for trial end event (default = 11)
        end_offset : int
            Offset in ms from end_code to use for binning (default = 1000)
        total_length : int
            Total length for trial (default = 2500 ms)
        cutoff : bool
            If True, pad matrix with NaNs after start/end codes (default = False)
        fs : int
            Sampling frequency (default = 30000 Hz)

    Outputs:
        binned_spikes : ndarray
            Spike count matrix with shape (num_channels x num_bins)
    '''
    
    # Get spike times from 'dat'
    spiketimes = np.cumsum(dat['spiketimesdiff']) + dat['firstspike']
    spiketimes = np.insert(spiketimes, 0, dat['firstspike'])

    # Filter spike events based on 'filter_spike_codes'
    valid_spikes = np.isin(dat['spikeinfo'][:, 1], filter_spike_codes)
    spiketimes_filtered = spiketimes[valid_spikes] / fs  # Select valid spikes + Convert to seconds
    spikechannels = dat['spikeinfo'][valid_spikes, 0]
    
    # Map spike channels to indices in the 'channels' list
    channel_map = {ch: idx for idx, ch in enumerate(channels)}  # Mapping channel numbers to indices
    valid_channels = np.isin(spikechannels, channels)  # Filter spikes to keep only those from channels in 'channels'
    
    spiketimes_filtered = spiketimes_filtered[valid_channels]
    spikechannels = spikechannels[valid_channels]
    
    # Initialize the binned_spikes matrix with rows corresponding to the channels in 'channels'
    num_channels = len(channels)
    num_bins = int(total_length / bin_size)
    binned_spikes = np.zeros((num_channels, num_bins))
    
    # Align spikes to the start or end of trial
    if align_type == 'start':
        # Get the start time for each trial
        start_time = (dat['trialcodes'][dat['trialcodes'][:, 1] == start_code, 2] + start_offset)
        
        # Select spikes within the desired time window
        window_start = start_time
        window_end = (start_time + total_length)
        relevant_spikes = (spiketimes_filtered >= window_start) & (spiketimes_filtered < window_end)
        relevant_channels = spikechannels[relevant_spikes]
        relevant_times = spiketimes_filtered[relevant_spikes]

        for channel in np.unique(relevant_channels):
            # Get spike times for each channel
            spike_times_for_channel = relevant_times[relevant_channels == channel]
            # Convert spike times to bin indices
            spike_bins = np.floor((spike_times_for_channel - window_start) / bin_size).astype(int)
            # Map channel to its index in 'channels'
            channel_idx = channel_map[channel]
            # Increment the corresponding bins for the channel
            for bin_idx in spike_bins:
                if 0 <= bin_idx < num_bins:
                    binned_spikes[channel_idx, bin_idx] += 1
            
        if cutoff:
            # Set bins after end_code to NaN
            end_times = (dat['trialcodes'][dat['trialcodes'][:, 1] == end_code, 2] + end_offset)
            for end_time in end_times:
                end_bin = np.floor((end_time - start_time) / bin_size).astype(int)
                if 0 <= end_bin < num_bins:
                    binned_spikes[:, end_bin + 1:] = np.nan
           
    elif align_type == 'end':
        # Get the end time for each trial
        end_time = (dat['trialcodes'][dat['trialcodes'][:, 1] == end_code, 2] + end_offset)
        
        # Select spikes within the desired time window
        window_start = end_time - total_length
        window_end = end_time
        relevant_spikes = (spiketimes_filtered >= window_start) & (spiketimes_filtered < window_end)
        relevant_channels = spikechannels[relevant_spikes]
        relevant_times = spiketimes_filtered[relevant_spikes]

        for channel in np.unique(relevant_channels):
            # Get spike times for each channel
            spike_times_for_channel = relevant_times[relevant_channels == channel]
            # Convert spike times to bin indices
            spike_bins = np.floor((spike_times_for_channel - window_start) / bin_size).astype(int)
            # Map channel to its index in 'channels'
            channel_idx = channel_map[channel]
            # Increment the corresponding bins for the channel
            for bin_idx in spike_bins:
                if 0 <= bin_idx < num_bins:
                    binned_spikes[channel_idx, bin_idx] += 1
        
        if cutoff:
            # Set bins before start_code to NaN
            start_times = (dat['trialcodes'][dat['trialcodes'][:, 1] == start_code, 2] + start_offset)
            for start_time in start_times:
                start_bin = np.floor((start_time - end_time) / bin_size).astype(int)
                if 0 <= start_bin < num_bins:
                    binned_spikes[:, :start_bin] = np.nan

    
    return binned_spikes



# For stitched data
def bin_stitched_data(trial_codes, activity_matrix, bin_size=1, align_type='start', start_code=70, start_offset=-500, end_code=11,
               end_offset=1, total_length=2500, cutoff=False):
    '''
    Bin spikes (works for smith lab data)

    Parameters: 
        trial_codes : dict
            
        bin_size : float
            Size of bin in ms (default = 1ms)
        
        align_type : str
            'start' or 'end'. Specifies how to align bins (default = 'start')
        start_code : int
            Code for trial start event (default = 70)
        start_offset : int
            Offset in ms from start_code to use for binning (default = -500)
        end_code : int
            Code for trial end event (default = 11)
        end_offset : int
            Offset in ms from end_code to use for binning (default = 1000)
        total_length : int
            Total length for trial (default = 2500 ms)
        cutoff : bool
            If True, pad matrix with NaNs after start/end codes (default = False)
       

    Outputs:
        binned_spikes : ndarray
            Spike count matrix with shape (num_channels x num_bins)
    '''
    
    
    
    # Initialize the binned_spikes matrix with rows corresponding to the channels in 'channels'
    
    num_latents, num_timepoints = activity_matrix.shape
    # Align spikes to the start or end of trial
    if align_type == 'start':
        # Get the start time for each trial
        start_time = (trial_codes[trial_codes[:, 1] == start_code, 2] - trial_codes[0, 2])*1000 + start_offset
        
        # Calculate the number of bins
        num_bins = int(total_length // bin_size)
        if total_length % bin_size != 0:
            num_bins += 1  # Handle the case where there's a remainder

        # Initialize an array to store binned data
        binned_activity = np.zeros((num_latents, num_bins))
        # Loop through each bin and compute the average activity across time points in that bin
        for i in range(num_bins):
            start_idx = int(i * bin_size + start_time)
            end_idx = int(min((i + 1) * bin_size, num_timepoints) + start_time)

            # Compute the mean across the timepoints in this bin
            binned_activity[:, i] = np.mean(activity_matrix[:, start_idx:end_idx], axis=1)
            
        if cutoff:
            # Set bins after end_code to NaN
            end_times = (trial_codes[trial_codes[:, 1] == end_code, 2] - trial_codes[0, 2] )*1000 + end_offset
            for end_time in end_times:
                end_bin = np.floor((end_time - start_time) / bin_size).astype(int)
                if 0 <= end_bin < num_bins:
                    binned_activity[:, end_bin + 1:] = np.nan

    elif align_type == 'end':
        # Get the end time for each trial
        end_time = (trial_codes[trial_codes[:, 1] == end_code, 2] - trial_codes[0,2])*1000 + end_offset
        start_time = end_time-total_length
        # Calculate the number of bins
        num_bins = total_length // bin_size
        if total_length % bin_size != 0:
            num_bins += 1  # Handle the case where there's a remainder

        # Initialize an array to store binned data
        binned_activity = np.zeros((num_latents, num_bins))
        # Loop through each bin and compute the average activity across time points in that bin
        for i in range(num_bins):
            start_idx = int(i * bin_size + start_time)
            end_idx = int(min((i + 1) * bin_size, num_timepoints) + start_time)

            # Compute the mean across the timepoints in this bin
            binned_activity[:, i] = np.mean(activity_matrix[:, start_idx:end_idx], axis=1)

        if cutoff:
            # Set bins before start_code to NaN
            start_times = (trial_codes[trial_codes[:, 1] == start_code, 2] - trial_codes[0,2])*1000 + start_offset
            for start_time in start_times:
                start_bin = np.floor((start_time - end_time) / bin_size).astype(int)
                if 0 <= start_bin < num_bins:
                    binned_activity[:, :start_bin] = np.nan
        
    return binned_activity



