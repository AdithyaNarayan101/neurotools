'''
Module to analyze kinematics of trajectories (Joystick, Arm, Eyes) 

Compute Reaction Times, Peak Speed, Velocity Profiles from (x,y,t) trajectory data

'''
import numpy as np

def find_chosen_target(traj,targets):
    '''
    Function to find the target that was closest to the 
    '''
    dist=targets.copy()
    for targ in targets:
        dist[targ]=(np.sqrt( (traj[0]-targets[targ][0])**2 + (traj[1]-targets[targ][1])**2))
    chosen_target = min(dist, key=dist.get)
    return chosen_target


def butter_lowpass_filter(data, cutoff, fs, order=2):
    from scipy.signal import butter,filtfilt
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
def compute_kinematics(trajectory, low_pass_filter='butter',low_pass_threshold=10, velocity_threshold = 0.2, threshold_type='trialwise', fs = 1000, plot_diagnostic=False):
    '''
    Takes in trajectories (joystick/eye/arm) and outputs kinematics like Response Time, Velocity Profiles, Peak Velocity (in ms and corresponding units)
    '''

    dist_from_center = np.sqrt(trajectory[0,:]**2 + trajectory[1,:]**2)

    if(low_pass_filter == 'butter'):
        dist_from_center_filtered = butter_lowpass_filter(dist_from_center, low_pass_threshold, fs) # low pass filter
    else:
        raise ExceptionType("Error message")

    vel = np.diff(dist_from_center_filtered) * fs / 1000 # px per ms, velocity

    # Unsure what the next two lines are doing - must test! (Why interpolate?)
    oldx = np.arange(len(vel)) + (fs / 1000 / 2)
    newx = np.arange(len(vel))
    interp_vel = np.interp(newx, oldx, vel)
    peak_vel = np.max(interp_vel)
    if(threshold_type=='trialwise'):
        velocity_threshold_trial = peak_vel*velocity_threshold # Threshold for determining RT set separately for each trial
    elif(threhold_type=='common'):
        velocity_threshold_trial = velocity_threshold # This threshold should've been computed across trials

    try:
        response_time = np.argwhere(interp_vel > velocity_threshold_trial)[0][0] / fs * 1000 # ms, locked to target onset
        peak_vel_time =    np.argwhere(interp_vel == peak_vel)[0][0] / fs * 1000 # ms, locked to target onset
    except:
        response_time = np.nan
        peak_vel_time = np.nan
        print("Warning: Velocity did not cross threshold. RT not stored")
    kinematics = {'Response Time':response_time, 'Peak Velocity':peak_vel, 'Peak Velocity Time':peak_vel_time, 'Velocity Profile':interp_vel }

    if plot_diagnostic: # Plots velocity profile + RT for this trial
        fig=go.Figure(go.Scatter(x=np.arange(0,len(interp_vel)),y=interp_vel))
        fig.add_trace(go.Scatter(x=[response_time,response_time],y=[np.min(interp_vel),np.max(interp_vel)],mode='lines',line=dict(width=4,color='black')))
        fig.show()

    return kinematics