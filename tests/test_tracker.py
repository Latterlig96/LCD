import cv2
import pytest
from tracker import Tracker


class TestTracker: 

    def test_valid_tracker_list(self): 
        tracker = Tracker(tracker_type="KCF")
        assert isinstance(tracker, Tracker)
    
    @pytest.mark.parametrize("tracker_type", [50, {"Invalid_Type": "Invalid_Type"}, None])
    def test_invalid_tracker_init(self, tracker_type):
        with pytest.raises(ValueError):
            tracker = Tracker(tracker_type=tracker_type)
            assert not isinstance(tracker, Tracker)
    
    def test_load_tracker_from_config(self, config_load):
        config = config_load
        tracker = Tracker.load_tracker_from_config(config.tracker_type)
        assert isinstance(tracker, Tracker)
    
    @pytest.mark.parametrize("tracker_type", [1, {"Invalid": "Invalid_type"}, None])
    def test_invalid_load_tracker_from_config(self, tracker_type):
        with pytest.raises(ValueError):
            tracker = Tracker.load_tracker_from_config(tracker_type)
            assert not isinstance(tracker, Tracker)
    
    @pytest.mark.parametrize("tracker_type", ["MIL", "TLD", "KCF", "BOOSTING", "MEDIANFLOW", "CSRT", "MOSSE"])
    def test_init_valid_tracker_type(self, tracker_type):
        tracker = Tracker(tracker_type=tracker_type)
        assert isinstance(tracker, Tracker)
        model = tracker.init_tracker()
    
    @pytest.mark.parametrize("tracker_type", ["MILD", "TILT", "KFC", "BOOSTER", "MEDIANFLAW", "CSRF", "MOUSE"])
    def test_init_invalid_tracker_type(self, tracker_type):
        tracker = Tracker(tracker_type=tracker_type)
        assert isinstance(tracker, Tracker)
        model = tracker.init_tracker()
        assert model is None
