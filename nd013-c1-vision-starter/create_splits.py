import argparse
import glob
import os
import random
import shutil
import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    print("*********************************")
    print(data_dir)
    
    val_ratio = 0.20
    test_ratio = 0.20
    
    src = data_dir  # Folder to copy images from

    allFileNames = os.listdir(src)
    print(allFileNames)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                                  [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                                   int(len(allFileNames) * (1 - val_ratio)),
                                                                   ])

    train_FileNames = [src + '//' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '//' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '//' + name for name in test_FileNames.tolist()]
    print(train_FileNames)
    print("************")

    print('Total images: '+ str(len(allFileNames)))
    print('Training: '+ str(len(train_FileNames)))
    print('Validation: '+  str(len(val_FileNames)))
    print('Testing: '+ str(len(test_FileNames)))

    root_dir = '/app/project/data'
    # # Creating Train / Val / Test folders (One time use)
    os.makedirs(root_dir + '/train/')
    os.makedirs(root_dir + '/val/' )
    os.makedirs(root_dir + '/test/')

    # Copy-pasting images
    for name in train_FileNames:
        shutil.move(name, root_dir + '/train/')

    for name in val_FileNames:
        shutil.move(name, root_dir + '/val/')

    for name in test_FileNames:
        shutil.move(name, root_dir + '/test/')

    # TODO: Implement function
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
