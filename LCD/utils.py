from typing import List, Tuple
import numba 
import numpy as np

__all__ = ["get_box_dimesions", "post_process"]

@numba.jit(nopython=True)
def get_box_dimensions(matches: np.ndarray,
                       height: int, 
                       width: int,
                       use_opencv: bool) -> Tuple[List[List[int]],
                                                  List[float],
                                                  List[int]]:
    """
        Function responsible for computing bounding box\n
        coordinates, confidence for this bounding box and\n
        a class ID
        Arguments: 
        ------------------------
        outputs: List - YOLO outputs list for detected objects\n
        height: int - frame height\n
        width: int - frame width\n
        Returns: 
        ------------------------
        boxes - List[List[int]] - bounding_box coordinates\n
        confs - List[float] - confidence for bounding boxes\n
        class_ids - List[int] - class ID for given bounding box\n
    """
    boxes, confs, class_ids = [], [], []
    for detect in matches: 
        scores = detect[5:] if use_opencv else detect[4:]
        class_id = np.argmax(scores)
        conf =scores[class_id]
        center_x = int(detect[0] * width)
        center_y = int(detect[1] * height)
        w = int(detect[2] * width)
        h = int(detect[3] * height)
        x = int(center_x - w/2)
        y = int(center_y - h/2)
        boxes.append([x, y, w, h])
        confs.append(float(conf))
        class_ids.append(class_id)
    
    return boxes, confs, class_ids 

def post_process(boxes: List[List[int]],
                 class_ids: List[int],
                 indexes: Tuple[int]
                 ) -> Tuple[List[List[int]],
                            List[int]]:
    """
        Post process bounding boxes after NMS operation
        Arguments: 
        ----------------------
        boxes - List[List[int]] - bounding box coordinates.\n
        class_ids - List[int] - class ID for given bounding box.\
        indexes - Tuple[int] - unique indexes for bounding boxes filtered after NMS operation.
        Returns: 
        ----------------------
        boxes - List[List[int]] - filtered bounding box coordinates\n
        class_ids - List[int] - filtered classes ID's for given bounding box\n
    """
    boxes = tuple(boxes[int(i)] for i in indexes)
    class_ids = tuple(class_ids[int(i)] for i in indexes)
    return boxes, class_ids
