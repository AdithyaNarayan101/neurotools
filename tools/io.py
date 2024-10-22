'''
Script with functions for input-output operations.

'''

def get_experiment_files(task,subject,data_path):
    subject_name = subject[0:2]
    dataset_path = data_path+task+"/";
    experiment_files = []
    experiment_files_path = []
    for root, dirs, files in os.walk(dataset_path):
        for file in sorted(files):
            if subject_name in file:  
                experiment_files.append(file)
                experiment_files_path.append(root)
    assert len(experiment_files)!=0
    print("There are " + str(len(experiment_files)) + " sessions: \n")
    #print(experiment_files)
    return experiment_files, experiment_files_path


# Only relevant for smith lab datasets
def load_trial_codes(trial_codes_path):
    import scipy.io as sio
    mat = sio.loadmat(trial_codes_path)
    struct,arr_out = {}, [[]] * (255+1)
    arr = mat['codesArray'][0]
    for i,code in enumerate(arr):
        code = code[0]
        if len(code) > 0:
            struct[code] = i+1
            arr_out[i+1] = (code)
    return struct,arr_out