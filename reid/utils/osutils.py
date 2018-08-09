from __future__ import absolute_import
import os
import errno


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def set_paths(paths_file=None, verbose=True):
    if not paths_file:
        raise ValueError('Set paths file')
    print('Set os.envirionments from {}...'.format(paths_file))
    if verbose: print('----------------------------------')
    paths_dict = {v.split('=')[0]:v.split('=')[1] for v in  open(paths_file,'r').read().split()}
    for path, v in paths_dict.items():
        os.environ[path] = v
        if verbose: print(path, v)
    if verbose: print('----------------------------------')
    print('done!')
