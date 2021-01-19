"""
Module with collection of utility functions used for SPADE analysis and
plotting
"""
import numpy as np
import os
import yaml


# Function to create new folders
def mkdirp(directory):
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass


# Function to split path to single folders
def split_path(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    folders.reverse()
    return folders


def split_all(path):
    """
    The function breaks out all of parts of a file or directory path
    """
    print('warning: the check depends on the file param_dict.npy \n')
    all_parts = []
    while 1:
        parts = os.path.split(path)
        # absolute path case
        if parts[0] == path:
            all_parts.insert(0, parts[0])
            break
        # relative path case
        elif parts[1] == path:
            all_parts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            all_parts.insert(0, parts[1])
    return all_parts


def check_tree_folder_structure(path):
    """
    The function checks if the tree folder structure of 'results' contains
    all the files required by the analysis, and returns the specifics of the
    missing files, if any.
    The module checks if all results are present in the folder, with the
    correct tree structure.
    If files are missing, it returns 0 and prints the respective paths in which
    they should be, and their characteristics.
    If the folder tree structure is complete, the function returns 1 and a
    message.

    Parameters:
         path : str.
            path of the results folder

    """
    print('warning: the check depends on the file param_dict.npy \n')
    # load parameter dictionary
    param_dict = np.load('param_dict.npy').item()
    with open("configfile.yaml", 'r') as stream:
        configfile = yaml.load(stream)
    sessions = configfile['sessions']
    # set of all combination of epochs and trial types
    epochs = set(param_dict[str(sessions[0])].keys())
    # constructing set and list with all job numbers
    set_fim_folders = set()
    list_fim_folders = []
    flag_annotations = 1
    flag_filtered_res = 1
    flag_fim_folders = 1
    flag_patt_time_hist = 1
    flag_complete = 1
    for key1, value1 in param_dict.items():
        for key2, value2 in value1.items():
            jobs_within_epoch = []
            for key3, value3 in value2.items():
                jobs_within_epoch.append(str(key3))
                list_fim_folders.append(str(key3))
            set_fim_folders.add(tuple(jobs_within_epoch))
    # walking the tree folder structure
    for dirName, subdirList, fileList in os.walk(path):
        # check if the session level is complete
        if subdirList == sessions:
            if not os.path.exists(dirName + 'patt_time_hist.npy'):
                print('patt_time_hist.npy file missing in %s: \n' % dirName)
                flag_patt_time_hist = 0
        # check if the epochs folder is complete
        if os.path.basename(dirName) in epochs:
            if not os.path.exists(dirName + '/annotations.npy'):
                print('annotations.npy file missing in: %s \n' % dirName)
                flag_annotations = 0
            if not os.path.exists(dirName + '/filtered_res.npy'):
                print('filtered_res.npy file missing in: %s \n' % dirName)
                flag_filtered_res = 0
        # check if the job number level is complete
        if os.path.basename(dirName) in list_fim_folders:
            if not os.path.exists(dirName + '/results.npy'):
                print('results.npy file missing in: %s' % dirName)
                # refer the parameters of the missing job through the
                # param_dict file
                print('the missing results characteristics are:')
                split_path = split_all(dirName)
                missing_file = param_dict[split_path[-3]][
                    split_path[-2]][int(split_path[-1])]
                print(missing_file, '\n')
                flag_fim_folders = 0
    flag = flag_complete * flag_annotations * flag_filtered_res\
        * flag_fim_folders * flag_patt_time_hist
    if flag:
        print('folder tree structure is complete')
    return flag


def get_position(el_id):
    """
    get the (x, y) position of an electrode on the Utah grid
    """
    return np.array([(el_id - 1) % 10, (el_id - 1) // 10])


def get_distance(el1, el2):
    """
    get the distance between two electrodes (in electrode distance units)
    on the Utah grid
    """
    pos1 = get_position(el1)
    pos2 = get_position(el2)
    return np.sqrt(np.sum((pos1 - pos2) ** 2))
