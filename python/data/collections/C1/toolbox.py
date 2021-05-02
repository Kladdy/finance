import sys
import os
sys.path.append('../../')
import log

logger = log.Log("DEBUG")

def mkdir(folder_name):
    """Makes sure a folder exists, otherwise creates it"""

    if not os.path.exists(folder_name):
        logger.INFO(f"Folder did not exist, creating {folder_name}...")
        os.makedirs(folder_name)