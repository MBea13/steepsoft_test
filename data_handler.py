import sys
import os
import pandas as pd


LABELS = {
    "folder_name" : "labels",
    "type_name" : "labeled_sleep",
    "separator" : " "
}
HEART_RATES = {
    "folder_name" : "heart_rate",
    "type_name" : "heartrate",
    "separator" : ","
}
MOTIONS = {
    "folder_name" : "motion",
    "type_name" : "acceleration",
    "separator" : " "
}
STEPS = {
    "folder_name" : "steps",
    "type_name" : "steps",
    "separator" : ","
}
DATA_FOLDER = sys.path[0] + "/data"
INNER_FOLDER_STRUCTURE = "{}/{}" #.format(DATA_FOLDER, NAMINGS['folder_name']])
PREPARED_DATA_FOLDER = sys.path[0] + "/prepared_data"
PATH_STRUCTURE = "{}/{}/{}_{}.txt" #.format(DATA_FOLDER, NAMINGS['folder_name'], idx, NAMINGS['type_name'])


def get_data_type(type):
    """Matches const dict for the given feature set

    Parameters
    ----------
    type : str
        feature set type (motions/steps/labels/heart_rates)

    Returns
    -------
    const dict
        const dict of given feature set

    Raises
    ------
    KeyError
        If the given type is not found
    """
    if type == "motions":
        return MOTIONS
    elif type == "steps":
        return STEPS
    elif type == "labels":
        return LABELS
    elif type == "heart_rates":
            return HEART_RATES
    else:
        raise KeyError("{} feature set type not found in [motions, steps, labels, heart_rates]".format(type))


def check_dir(path):
    """Checks if a directory exists and if not creates it

    Parameters
    ----------
    path : str
        path to directory
    """
    found = os.path.isdir(path)
    if not found: 
        os.makedirs(path) 
    return found
        
        
def save_data_frame(type_const, df, id):
    try:
        data_type = get_data_type(type_const)
    except KeyError as e:
        print(e)
        return None
    check_dir(PREPARED_DATA_FOLDER)
    file_w_path = PATH_STRUCTURE.format(PREPARED_DATA_FOLDER, data_type["folder_name"], id, data_type["type_name"])
    check_dir(INNER_FOLDER_STRUCTURE.format(PREPARED_DATA_FOLDER, data_type["folder_name"]))
    df.to_csv(file_w_path, sep=data_type["separator"], header=False)
        
    
def get_subject_ids():
    """Defines all subject ids

    Returns
    -------
    list of ints
        subject indices
    """
    directory = DATA_FOLDER + "/" + LABELS["folder_name"]
    ids = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            ids.append(int(f.split('/')[-1].split('_')[0]))
    return ids


def read_single_file(type_const, idx, use_prepared):
    if not use_prepared:
        file_w_path = PATH_STRUCTURE.format(DATA_FOLDER, type_const["folder_name"], idx, type_const["type_name"])
    else:
        file_w_path = PATH_STRUCTURE.format(PREPARED_DATA_FOLDER, type_const["folder_name"], idx, type_const["type_name"])
    return pd.read_csv(file_w_path, sep=type_const["separator"], header=None).astype("float")
    
    
def read_files(type, idx=None, use_prepared=False):
    """Reads all files of the given feature set

    Parameters
    ----------
    type : str
        feature set type (motions/steps/labels/heart_rates)
    idx : int, optional
        subject index (if we want to read only a given feature set), if None all subject features will be loaded, by default None
    use_prepared : bool, optional
        use True if you want to use the already prepared data, by default False

    Returns
    -------
    dict
        Loaded data with its subject ids as keys

    Raises
    ------
    KeyError
        If the given type is not found
    """
    ret_dict = {}
    subject_ids = get_subject_ids()
    try:
        data_type = get_data_type(type)
    except KeyError as e:
        print(e)
        return None
    
    if idx is not None:
        return read_single_file(data_type, idx, use_prepared)
    for idx in subject_ids:
        ret_dict[idx] = read_single_file(data_type, idx, use_prepared)
    return ret_dict



