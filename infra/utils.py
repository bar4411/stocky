import gzip
import json
import os
import pickle


def save_to_pickle(obj, file_path):
    make_folder_for_file_creation_if_not_exists(file_path)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def make_folder_for_file_creation_if_not_exists(file_path):
    folder_path = os.path.dirname(file_path)
    if folder_path:
        make_folder(folder_path)


def is_file_exist(file_path):
    return os.path.isfile(file_path)


def make_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

def load_json(file_path):
    with open(file_path) as f:
        json_file = json.load(f)
        f.close()
    return json_file

def load_pickle(file_path):
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as handle:
            pkl_file = pickle.load(handle)
    else:
        with open(file_path, 'rb') as handle:
            pkl_file = pickle.load(handle)
        handle.close()
    return pkl_file