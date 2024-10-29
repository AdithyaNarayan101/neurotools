'''
Module with functions for input-output operations.

'''

# Import stuff
from scipy.io import loadmat as tradLoadmat
import h5py 
import numpy as np
import os
import mat73
import pandas as pd
# Function definitions 


def read_txt_file(file_name):
    import ast

    with open(file_name, 'r') as file:
        data = file.read()
        data_dict = ast.literal_eval(data)
    return data_dict

def load_concat_subject_data(subject,all_dates,data_path,data_type='behav'):
    # Load each session's data and concatenate into a single dataframe: 
    df=[]
    for date in all_dates:
        df.append(pd.read_csv(data_path+'df_'+data_type+'_'+date+'.csv'))
    
    df=pd.concat(df,ignore_index=True)
    
    return df


def load_trial_codes(trial_codes_path):

    '''
    Loads ex-globals trial codes; see https://github.com/SmithLabNeuro/Ex/blob/master/ex_control/exGlobals.m
    
    Only relevant for smith lab datasets

    '''
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


def get_experiment_file(data_path, subject='Sa', task='peripheralFocus', date='2222222',area='M1'):

    '''
    Find the file in the data_path that matches the query(subject,task,area,date)

    '''
    subject_name = subject[0:2].lower()
    date_short = date[2:]
    query={subject_name,task,date_short,area}
    experiment_files = []
    experiment_files_path = []
    for item in os.listdir(data_path):
        item_path=data_path+item
        if os.path.isdir(item_path):
             continue
        if(all(i.lower() in item.lower() for i in query)):
            experiment_files.append(item)

    assert len(experiment_files)==1
    
    return experiment_files[0]


## Functions needed to load mat files into python

def LoadHdf5Mat(matfilePath):
    # def LookAheadDeref(hdf5File, reference):
        
    def UnpackHdf5(hdf5Matfile, hdf5Group):
        out = {}
        if type(hdf5Group) is h5py._hl.group.Group:
            for key in hdf5Group:
                out[key] = hdf5Group[key]
                if type(out[key]) is h5py._hl.group.Group:
                    out[key] = UnpackHdf5(hdf5Matfile,out[key])
                elif type(out[key]) is h5py._hl.dataset.Dataset:
                    out[key] = UnpackHdf5(hdf5Matfile,out[key])
                elif type(out[key]) is h5py.h5r.Reference:
                    out[key] = UnpackHdf5(hdf5Matfile,out[key])
                        
                
                    
                    
#                        valsLambda = (lambda hMF=hdf5Matfile, hG=hdf5Group, k=key: [hMF[hG[k][row, col]] for row in range(hG[k].shape[0]) for col in range(hG[k].shape[1])])
#                        valsList = list(valsLambda())
#                        for idx, val in enumerate(valsList):
#                            if type(val) is h5py._hl.group.Group:
#                                valsList[idx] = UnpackHdf5(hdf5Matfile,val)
#                            elif 'MATLAB_empty' in val.attrs.keys(): # deal with empty arrays
#                                valsList[idx] = np.ndarray(0)
#                            elif np.any([type(nestedVal) is h5py.h5r.Reference for nestedVal in val[()][0]]):
#                                valsList[idx] = np.ndarray(val.shape)
#                                for row in range(val.shape[0]):
#                                    for col in range(val.shape[1]):
#                                        #if valsList[idx] = val[()]
#                                        if type(val[()][row, col]) is h5py.h5r.Reference:
#                                            valsList[idx][row,col] = UnpackHdf5(hdf5Matfile,hdf5Group[val[()][row,col]])
#                                        else:
#                                            valsList[idx][row,col] = val[()][row,col]
#                            else:
#                                valsList[idx] = val[()]
#                        out[key] = 5
        elif type(hdf5Group) is h5py._hl.dataset.Dataset:
            out = np.ndarray(hdf5Group.shape, dtype=object)
            
            if hdf5Group.dtype == np.dtype('object'):
                
#                out = np.frompyfunc(list, 0, 1)(np.empty(hdf5Group.shape, dtype=object))
                with np.nditer([out, hdf5Group], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                    #while not iterRef.finished:
                    for valOut, valIn in iterRef:
                        if type(valIn[()]) is h5py._hl.group.Group:
                            valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
    #                            valOut[()].append(UnpackHdf5(hdf5Matfile, valIn[()]))
                        elif type(valIn[()]) is h5py._hl.dataset.Dataset:
                            valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
    #                            valOut[()].append(UnpackHdf5(hdf5Matfile, valIn[()]))
                        elif type(valIn[()]) is h5py.h5r.Reference:
                            valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
    #                            valOut[()].append(UnpackHdf5(hdf5Matfile, valIn[()]))
                        else:
                            valOut[()] = valIn[()]
    #                            valOut[()].append(valIn[()])
                            
                        #iterRef.iternext()
                      
                   
                    out = iterRef.operands[0]
                    out = out.T # undo Matlab's weird transpose when saving...
#                for row in range(out.shape[0]):
#                    for col in range(out.shape[1]):
#                       #if valsList[idx] = val[()]
#                       if type(out[row, col]) is h5py.h5r.Reference:
#                           out[row,col] = UnpackHdf5(hdf5Matfile, out[row,col])
#                       elif type(out[row, col]) is h5py._hl.group.Group:
#                           out[row,col] = UnpackHdf5(hdf5Matfile, out[row,col])
#                       else:
#                           out[row,col] = out[row,col]
            else:
                # apparently type dataset can also store arrays like type
                # reference and I just give up
                #
                # but I'm also renaming this variable to parallel what was done
                # for the reference and perhaps someday I'll make it its own
                # function
                deref = hdf5Group
                if 'MATLAB_empty' in deref.attrs.keys(): # deal with empty arrays
                    # print('empty array')
                    # print(deref[()])
                    if 'MATLAB_class' in deref.attrs.keys():
                        pass
                        # print(deref.attrs['MATLAB_class'])
                    out = np.ndarray(0)
                    return out.T
                
                if 'MATLAB_int_decode' in deref.attrs.keys():
                    if 'MATLAB_class' in deref.attrs.keys():
                        if deref.attrs['MATLAB_class'] == b'char':
                            out = "".join([chr(ch) for ch in deref[()]])
                            return out
                        elif deref.attrs['MATLAB_class'] == b'logical':
                            pass # uint8, the default, is a fine type for logicals
                        else:
                            # print(deref.attrs['MATLAB_class'])
                            # print('int decode but class not char...')
                            pass
                    else:
                        # print('int decode but no class?')
                        pass
                
                out = deref[()]
                out = out.T # for some reason Matlab transposes when saving...
        elif type(hdf5Group) is h5py.h5r.Reference:
            deref = hdf5Matfile[hdf5Group]
            
            if type(deref) is h5py._hl.group.Group:
                out = UnpackHdf5(hdf5Matfile, deref)
            elif deref.dtype == np.dtype('object'):
                try:
                    out = np.ndarray(deref.shape, dtype=object)
                except (AttributeError) as err:
                    raise RuntimeError('problem with forming iterator of a non-group a')
                        
                else:
    #                    with np.nditer(out, ['refs_ok'], ['readwrite']) as iterRef:
                    with np.nditer([out, deref], ['refs_ok'], [['writeonly'], ['readonly']]) as iterRef:
                        for valOut, valIn in iterRef:
                            if type(valIn[()]) is h5py._hl.group.Group:
                                valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
                            elif type(valIn[()]) is h5py._hl.dataset.Dataset:
                                valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
                            elif type(valIn[()]) is h5py.h5r.Reference:
                                valOut[()] = UnpackHdf5(hdf5Matfile, valIn[()])
                            else:
                                # print('non-hdf5 object')
                                if 'MATLAB_empty' in deref.attrs.keys(): # deal with empty arrays
                                    valOut[()] = np.ndarray(0)
                                else:
                                    valOut[()] = valIn[()]
                        out = iterRef.operands[0]
                        out = out.T # undo Matlab's weird transpose when saving...
            else:
                if 'MATLAB_empty' in deref.attrs.keys(): # deal with empty arrays
                    # print('empty array')
                    # print(deref[()])
                    if 'MATLAB_class' in deref.attrs.keys():
                        # print(deref.attrs['MATLAB_class'])
                        pass
                    out = np.ndarray(0)
                    return out.T
                
                if 'MATLAB_int_decode' in deref.attrs.keys():
                    if 'MATLAB_class' in deref.attrs.keys():
                        if deref.attrs['MATLAB_class'] == b'char':
                            out = "".join([chr(ch) for ch in deref[()]])
                            return out
                        elif deref.attrs['MATLAB_class'] == b'logical':
                            pass # uint8, the default, is a fine type for logicals
                        else:
                            print(deref.attrs['MATLAB_class'])
                            print('int decode but class not char...')
                    else:
                        print('int decode but no class?')
                
                out = deref[()]
                out = out.T # for some reason Matlab transposes when saving...
#                except (AttributeError, TypeError) as err:
#                    if type(out) is h5py._hl.group.Group:
#                        out = np.asarray(UnpackHdf5(hdf5Matfile, deref))
#                    else:
#                        raise RuntimeError('problem with forming iterator of a non-group b')
                    
#            elif type(hdf5Group) is h5py.h5r.Reference:
#                out = hdf5Matfile[hdf5Group]
#                iterRef = np.nditer(out, ['refs_ok'])
#                for val in iterRef:
#                    if type(val[()]) is h5py._hl.group.Group:
#                        UnpackHdf5(hdf5Matfile, val[()])
#                    elif type(val[()]) is h5py._hl.dataset.Dataset:
#                        UnpackHdf5(hdf5Matfile, val[()])
#                    elif type(val[()]) is h5py.h5r.Reference:
#                        UnpackHdf5(hdf5Matfile, val[()])
#                    else:
#                        if 'MATLAB_empty' in out.attrs.keys(): # deal with empty arrays
#                            np.ndarray(0)
#                        else:
#                            val[()]
#                out = iterRef.operands[0]
        
        return out
    
    hdf5Matfile = h5py.File(matfilePath, 'r')
    
    out = {}

    # this loop looks very similar to that in unpacking the group, but it
    # specifically ignores the #refs# key... I'm also not sure its terminal
    # condition is quite right, as I don't know if a non-structure variable
    # is saved as a Dataset at the top of the hierarchy--I assume so?
    for key in hdf5Matfile:
        if key == '#refs#':
            pass
        
        elif type(hdf5Matfile[key]) is h5py._hl.group.Group:
            out[key] = UnpackHdf5(hdf5Matfile, hdf5Matfile[key])
        elif type(hdf5Matfile[key]) is h5py._hl.dataset.Dataset:
            out[key] = UnpackHdf5(hdf5Matfile, hdf5Matfile[key])
            
    return out
 
    
def load_mat_file(mat_file_path):
    # Emilio's code for loading mat files
    try:
        annots = tradLoadmat(mat_file_path)
    except (NotImplementedError, MemoryError):
        annots = LoadHdf5Mat(mat_file_path)
        
    return annots

def load_mat73(file_path):
    # Uses mat73 package to load mat files
    dat = mat73.loadmat(file_path, use_attrdict=True) 
    return dat


def load_session_dat(data_path,subject='Sa',area='M1',task='Focus',date='0000'):
    
    '''
    Calls get_experiment_files to get the appropriate session path + file name. Then calls load_mat_file

    '''
    session_file = get_experiment_file(data_path,subject=subject,area=area,task=task,date=date)

    try:
        dat = load_mat73(data_path+session_file)
    except:
        raise Exception("Failed to load dat file")
    
    return dat['dat']
