import numpy as np
import pytest
from utils import get_box_dimensions, post_process


def test_get_box_dimesinos():
    test_height = 600 
    test_width = 600 
    use_opencv=True
    for shape in (300, 1200, 4800):
        test_input = [np.random.randn(shape, 85) for i in range(5)]
        boxes, scores, class_ids = get_box_dimensions(test_input, test_height, test_width, use_opencv)

        assert isinstance(boxes, list)
        for box in boxes: 
            assert isinstance(box , list)
            for value in box:
                assert isinstance(value, (int, float))
        assert isinstance(scores, list)
        for score in scores: 
            assert isinstance(score, (int, float))
        assert isinstance(class_ids, list)
        for class_id in class_ids: 
            assert isinstance(class_id, (int, float))

@pytest.mark.parametrize("boxes,class_ids,indexes", [(np.random.randn(300, 85), np.arange(1, 80), np.arange(1, 79))])
def test_post_process(boxes, class_ids, indexes):
    boxes, class_ids = post_process(boxes, class_ids, indexes)
    assert isinstance(boxes, list)
    assert isinstance(class_ids, list)
