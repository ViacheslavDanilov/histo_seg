import base64
import io
import os
import zlib
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

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
    'Venule lumen': {
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
        'color': [150, 240, 52],
    },
    'Nerve trunks': {
        'id': 9,
        'color': [144, 19, 254],
    },
    # 'Cell nucleus': {
    #     'id': 10,
    #     'color': [150, 150, 150],
    # },
}

CLASS_COLOR = {
    class_name: tuple(class_info['color']) for class_name, class_info in CLASS_MAP.items()  # type: ignore
}

CLASS_ID = {class_name: class_info['id'] for class_name, class_info in CLASS_MAP.items()}

CLASS_ID_REVERSED = dict((v, k) for k, v in CLASS_ID.items())

METADATA_COLUMNS = [
    'image_path',
    'image_name',
    'slide',
    'tile',
    'dataset',
    'image_width',
    'image_height',
    'type',
    'x1',
    'y1',
    'x2',
    'y2',
    'box_width',
    'box_height',
    'area',
    'area_label',
    'encoded_mask',
    'class_id',
    'class_name',
]


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
) -> List[str]:
    """Get list of files with the specified extensions.

    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
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
                if file_ext in ext_list:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
    all_files.sort()
    return all_files


def convert_base64_to_numpy(
    s: str,
) -> np.ndarray:
    """Convert base64 encoded string to numpy array.

    import supervisely as sly
    encoded_string = 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'
    figure_data = sly.Bitmap.base64_2_data(encoded_string)
    print(figure_data)
    #  [[ True  True  True]
    #   [ True False  True]
    #   [ True  True  True]]
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)

    img_decoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if (len(img_decoded.shape) == 3) and (img_decoded.shape[2] >= 4):
        mask = img_decoded[:, :, 3].astype(bool)  # 4-channel images
    elif len(img_decoded.shape) == 2:
        mask = img_decoded.astype(bool)  # flat 2D mask
    else:
        raise RuntimeError('Wrong internal mask format')
    return mask


def convert_numpy_to_base64(
    mask: np.ndarray,
) -> str:
    """Convert numpy array to base64 encoded string.

    # Get Bitmap from annotation
    for label in ann.labels:
        if type(label.geometry) == sly.Bitmap:
            figure = label.geometry

    encoded_string = sly.Bitmap.data_2_base64(figure.data)
    print(encoded_string)
    # 'eJzrDPBz5+WS4mJgYOD19HAJAtLMIMwIInOeqf8BUmwBPiGuQPr///9Lb86/C2QxlgT5BTM4PLuRBuTwebo4hlTMSa44sKHhISMDuxpTYrr03F6gDIOnq5/LOqeEJgDM5ht6'
    """
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes_enc = bytes_io.getvalue()
    encoded_mask = base64.b64encode(zlib.compress(bytes_enc)).decode('utf-8')
    return encoded_mask
