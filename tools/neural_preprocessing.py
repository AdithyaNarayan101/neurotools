
'''
Preprocessing script for neural data. Performs binning of spikes for Smith lab data

'''

def bin_spikes(dat, bin_size=1, channels=list(range(64)), filter_spike_codes=[1], 
               align_type='start',start_code=70, start_offset = -500, end_code=11,
                end_offset=1000, total_length=2500, cutoff = False)

'''
Bin spikes (works for smith lab data)

***

Parameters: 

dat : 'dat' dict extracted from .mat file (after nev2dat)
bin_size: size of bin in ms (default = 1ms)
channels: list of channel numbers (default 0:63)
filter_spike_codes: 'sorting' code or nas sorting code. Default is [1] (can also be 0,1..,255)
align_type: Specifies if Binning should be done aligned to the 'start' code, 'end' code, or 'both'
start_code/end_code: Specified code to align bins across trials
start_offset/end_offset: Offset (in ms) with respect to the start/end codes
total_length: total length in ms for binning spikes
cutoff: if True, spike-count-matrix is nan-padded for bins  after end_code (if align=start) or before start_code (if align=end)

***

Outputs: 

binned_spikes: Spike Count Matrix (number of channels x number of bins)

'''


