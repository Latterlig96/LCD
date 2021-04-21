import numpy as np 
import pytest
from loader import Loader
import onnxruntime

class TestLoader:

    def test_valid_loader_init(self):
        loader = Loader(model_weights_path="../config/yolov3.weights",
                        model_config_path="../config/yolov3.cfg",
                        coco_names="../config/coco.names")
        assert isinstance(loader, Loader)
    
    def test_load_from_config(self, config_load): 
        config = config_load 
        loader = Loader.load_from_config(config)
        assert isinstance(loader, Loader)
    
    def test_load_cv2_yolo(self, config_load):
        config = config_load 
        loader = Loader.load_from_config(config) 

        assert isinstance(loader, Loader) 

        model, classes, output_layers, colors = loader.load_cv2_yolo()

        assert model is not None 
        assert isinstance(classes, list) 
        assert isinstance(output_layers, list)
        assert isinstance(colors, np.ndarray)
    
    def test_load_onnx_yolo(self, config_load):
        config = config_load 
        config.model_weights_path = "../config/yolov3.onnx"
        loader = Loader.load_from_config(config)

        assert isinstance(loader, Loader)

        session, classes, colors = loader.load_onnx_yolo()

        assert session is not None 
        assert isinstance(session, onnxruntime.InferenceSession) 
        assert isinstance(classes, list) 
        assert isinstance(colors, np.ndarray)
    
    def test_load_from_config_raise_error(self, config_load):
        config = config_load
        config.model_weights_path = None 
        with pytest.raises(ValueError): 
            loader = Loader.load_from_config(config)
            assert not isinstance(loader, Loader)
