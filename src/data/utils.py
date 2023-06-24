import os
from pathlib import Path
from typing import List, Union

CLASS_MAP = {
    'Arteriole lumen': {
        'id': 1,
        'color': [212, 0, 2],
    },
    'Arteriole media': {
        'id': 2,
        'color': [255, 124, 121],
    },
    'Arteriole adventitia': {
        'id': 3,
        'color': [227, 119, 194],
    },
    'Venula lumen': {
        'id': 4,
        'color': [31, 119, 180],
    },
    'Venule wall': {
        'id': 5,
        'color': [174, 199, 232],
    },
    'Capillary lumen': {
        'id': 6,
        'color': [105, 45, 33],
    },
    'Capillary wall': {
        'id': 7,
        'color': [196, 156, 148],
    },
    'Immune cells': {
        'id': 8,
        'color': [100, 100, 100],  # TODO: define a unique color
    },
    'Nerve trunks': {
        'id': 9,
        'color': [127, 127, 127],  # TODO: define a unique color
    },
    'Cell nucleus': {
        'id': 10,
        'color': [150, 150, 150],  # TODO: define a unique color
    },
}

CLASS_COLOR = {
    class_name: tuple(class_info['color']) for class_name, class_info in CLASS_MAP.items()  # type: ignore
}

CLASS_ID = {class_name: class_info['id'] for class_name, class_info in CLASS_MAP.items()}

CLASS_ID_REVERSED = dict((v, k) for k, v in CLASS_ID.items())

METADATA_COLUMNS = [
    'image_path',
    'Image name',
    'slide',
    'tile',
    'dataset',
    'image_width',
    'image_height',
    'x1',
    'y1',
    'x2',
    'y2',
    'xc',
    'yc',
    'box_width',
    'box_height',
    'box_area',
    'box_label',
    'area',
    'mask',
    'class_id',
    'class',
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
