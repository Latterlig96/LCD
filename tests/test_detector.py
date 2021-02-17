import pytest
from detector import Detector


class TestDetector: 

    def test_valid_detector_inti(self, config_load): 
        config = config_load 
        detector = Detector(config) 
        assert isinstance(detector, Detector)
    
    @pytest.mark.parametrize("config", [1, "config", None])
    def test_invalid_detector_init(self, config):
        with pytest.raises(ValueError):
            detector = Detector(config)
            assert not isinstance(detector, Detector)
    
    def test_prepare_detector_opencv_yolo(self, config_load):
        config = config_load
        detector = Detector(config)
        detector.prepare_detector()
        assert isinstance(detector, Detector) 
        assert hasattr(detector, 'model')
        assert hasattr(detector, 'classes')
        assert hasattr(detector, 'output_layers')
        assert hasattr(detector, 'colors')
    
    def test_prepare_detector_raise_error(self, config_load):
        config = config_load
        config.model_weights_path = None 
        detector = Detector(config)

        with pytest.raises(ValueError):
            detector.prepare_detector()
