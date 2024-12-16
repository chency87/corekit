# import yaml

# def load_yaml(file_path):
#     return yaml.safe_load(open(file_path))

from pathlib import Path
import json
from shutil import rmtree

def assert_folder(file_path):
    if not Path(file_path).exists():
        Path(file_path).mkdir(parents= True, exist_ok= True)
    return file_path

def rm_folder(folder_path):    
    rmtree(Path(folder_path), ignore_errors= True)

def reset_folder(folder_path):
    rm_folder(folder_path)
    assert_folder(folder_path)
    

def load_json(file_path):
    assert Path(file_path).exists(), f"file: {file_path} does not exists. "
    return json.load(open(file_path))

    
__all__ = ['load_json', 'assert_folder', 'rm_folder', 'reset_folder']