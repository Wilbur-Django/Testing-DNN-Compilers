import datetime
import os
import shutil

def clear_and_make_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def path_append_timestamp(save_dir):
    sub_dir_name = datetime.datetime.now().strftime("%m%d_%H%M%S")
    save_path = os.path.join(os.path.curdir, save_dir, sub_dir_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path

def norm_user_path(user_path):
    return os.path.expanduser(user_path)

def file_name_no_ext(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def get_ext(file_path):
    return os.path.splitext(os.path.basename(file_path))[1]

def change_ext(file_path, new_ext):
    file_name = file_name_no_ext(file_path)
    return os.path.join(os.path.dirname(os.path.abspath(file_path)), f"{file_name}.{new_ext}")
