import shutil
import os

def copy_list(file_list, destination):
    new_paths = []
    for file_path in file_list:
        if file_path:
            shutil.copy2(file_path,destination)
            new_paths.append(os.path.join(destination,file_path.split('/')[-1]))
        else:
            new_paths.append(None)
    return new_paths
