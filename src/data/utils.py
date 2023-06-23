import os
from pathlib import Path
from typing import List, Union

# TODO: change wall to media and adventitia
CLASS_MAP = {
    '': None,
    'Arteriole lumen': 1,
    'Arteriole media': 2,
    'Arteriole adventitia': 3,
    'Venula lumen': 4,
    'Venule wall': 5,
    'Capillary lumen': 6,
    'Capillary wall': 7,
    'Immune cells': 8,
    'Nerve trunks': 9,
}

CLASS_MAP_REVERSED = dict((v, k) for k, v in CLASS_MAP.items())

METADATA_COLUMNS = [
    'Image path',
    'Image name',
    'Slide',
    'Tile',
    'Dataset',
    'Image width',
    'Image height',
    'x1',
    'y1',
    'x2',
    'y2',
    'Box width',
    'Box height',
    'Box area',
    'Box label',
    'Area',
    'Mask',
    'Class ID',
    'Class',
]


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
    include_template: str = '',
) -> List[str]:
    """Get list of files with the specified extensions.

    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
        include_template: include directories with this template
    Returns:
        all_files: a list of file paths
    """
    all_files = []
    src_dirs = [src_dirs] if isinstance(src_dirs, str) else src_dirs
    ext_list = [ext_list] if isinstance(ext_list, str) else ext_list
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_ext = Path(file).suffix
                file_ext = file_ext.lower()
                dir_name = os.path.basename(root)
                if file_ext in ext_list and include_template in dir_name:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
    all_files.sort()
    return all_files
