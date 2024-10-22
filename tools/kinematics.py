'''
Functions to analyze kinematics of trajectories (Joystick, Arm, Eyes) 

'''

def compute_kinematics(trajectory, filter='butter',low_pass_threshold=10, velocity_threshold = 0.2, threshold_type='trialwise', fs = 1000, plot_diagnostic=False):
    '''
    Takes in trajectories (joystick/eye/arm) and outputs kinematics like Response Time, Velocity Profiles, Peak Velocity (in ms and corresponding units)
    '''

    dist_from_center = np.sqrt(trajectory[0,:]**2 + trajectory[1,:]**2)

    if(filter == 'butter'):
        dist_from_center_filtered = butter_lowpass_filter(dist_from_center, low_pass_threshold, fs) # low pass filter
    else:
        raise ExceptionType("Error message")

    vel = np.diff(dist_from_center_filtered) * fs / 1000 # px per ms, velocity

    # Unsure what the next two lines are doing - must test! (Why interpolate?)
    oldx = np.arange(len(vel)) + (fs / 1000 / 2)
    newx = np.arange(len(vel))
    interp_vel = np.interp(newx, oldx, dxdt)
    peak_vel = np.max(interp_vel)
    if(threshold_type=='trialwise'):
        velocity_thresh_trial = peak_vel*velocity_thresh # Threshold for determining RT set separately for each trial
    elif(threhold_type=='common'):
        velocity_thresh_trial = velocity_thresh # This threshold should've been computed across trials


    response_time = np.argwhere(interp_vel > velocity_thresh_trial)[0][0] / fs * 1000 # ms, locked to target onset
    peak_vel_time =    np.argwhere(interp_vel == peak_vel)[0][0] / fs * 1000 # ms, locked to target onset
        
    kinematics = {'Response Time':response_time, 'peak_velocity':peak_vel, 'peak_velocity_time':peak_vel_time, 'velocity_profile':interp_vel }

    if plot_diagnostic: # Plots velocity profile + RT for this trial
        fig=go.Figure(go.Scatter(x=np.arange(0,len(interp_vel)),y=interp_vel))
        fig.add_trace(go.Scatter(x=[response_time,response_time],y=[np.min(interp_vel),np.max(interp_vel)],mode='lines',line=dict(width=4,color='black')))
        fig.show()

return kinematics