# Utility functions for tests

import os

def get_data_folder():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(cur_dir, "data")
    return data_folder
