import logging 
from typing import TypeVar, Union, List, Tuple
import numpy as np
import cv2

_Tracker = TypeVar("Tracker")


class Tracker: 

    supported_trackers = [
        "BOOSTING",
        "MIL",
        "KCF",
        "TLD",
        "MEDIANFLOW",
        "MOSSE",
        "CSRT"
    ]

    logging.getLogger(__name__)
    
    def __init__(self, tracker_type: str):
        if not isinstance(tracker_type, str): 
            raise ValueError(
                f"Expected tracker type to be string, but got {type(tracker_type)} instead")
        self.tracker_type = tracker_type
    
    @classmethod 
    def load_tracker_from_config(cls,
                                 tracker_type: str):
        """
            Init tracker instance with defined tracker type
            Arguments: 
            -----------------------
            tracker_type - str - tracker_type defined in config file
            Returns: 
            -----------------------
            cls - Tracker objects class
        """
        if not isinstance(tracker_type, str):
            raise ValueError(
                    f"Tracker type must be string, but got {tracker_type} instead")
        return cls(tracker_type)
    
    def init_tracker(self,
                    multi_tracker: bool = False
                    ) -> Union[None, _Tracker]:
        """
            Function responsible for instancing specified \n
            tracker type (if it is in supported tracker types)
            Returns: 
            ------------------------
            None - function initialize instance of a specified tracker type
            and if provided, multi tracker instance
        """
        if self.tracker_type not in self.supported_trackers:
            logging.error(f""" Invalid tracker was provided, got {self.tracker_type},
                                but expected one of {self.supported_trackers}, please correct mistake
                                for program to execute
                            """)
            return
        if hasattr(self, f"init_{self.tracker_type.lower()}_tracker"):
            self.tracker = getattr(self, f"init_{self.tracker_type.lower()}_tracker")
            if self.tracker is not None: 
                logging.info(
                    f"Tracker type {self.tracker_type} initialized successfully"
                )
            if multi_tracker:
                self.multi_tracker = cv2.MultiTracker_create()
    
    def feed_init_frame(self,
                        multi_tracker: bool,
                        frame: np.ndarray,
                        bbox: Union[np.ndarray, List[np.ndarray]]
                        ) -> None:
        """
            Function is responsible for feeding initial frames for
            tracker with bounding boxes referring to tracked object\n
            NOTE: Function can be also used when updating new objects
            to tracker but only if you use multi_tracker otherwise
            function returns non supported error
            Arguments:
            -----------------------------
            multi_tracker: bool - whether to use multi tracker instance\n
            frame: np.ndarray - video frame\n
            bbox: Union[np.ndarray, List[np.ndarray]] - bounding boxes of objects to track\n
            Returns: 
            -----------------------------
            None
        """
        if bbox is None: 
            logging.info("No bounding boxes found to feed the tracker, operation terminated")
            return
        if multi_tracker: 
            if isinstance(bbox, tuple):
                for bb in bbox: 
                    self.multi_tracker.add(self.tracker, frame, tuple(bb))
            else: 
                self.multi_tracker.add(self.tracker, frame, tuple(bbox))
        else: 
            if isinstance(bbox, list): 
                logging.error(f"Cannot initialize {self.tracker_type} with list of bounding boxes")
            self.tracker.init(frame, bbox)
    
    def update_frames(self,
                      multi_tracker: bool,
                      frame: np.ndarray
                      ) -> Tuple[bool, Union[np.ndarray, List[np.ndarray]]]:
        """
            Update tracker with new frame
            Arguments: 
            -------------------------------
            multi_tracker: bool - whether to use multi tracker\n
            frame: np.ndarray - video frame\n
            Returns:
            -------------------------------
            success: bool - boolean indicating whether we have successfully obtained new bounding box
            bounding_box: Union[np.ndarray, List[np.ndarray]] - updated bounding box parameters
        """ 
        if multi_tracker: 
            success, bounding_box = self.multi_tracker.update(frame)
            return success, bounding_box
        success, bounding_box = self.tracker_update(frame)
        return success, bounding_box
    
    def init_boosting_tracker(self) -> _Tracker:
        """
            Initialize BOOSTING tracker instance
            Returns: 
            ----------------------------
            _Tracekr - tracker_instance
        """
        return cv2.TrackerBoosting_create()

    def init_mil_tracker(self) -> _Tracker:
        """
            Initialize MIL (Multiple Instance Learning) tracker instance
            Returns: 
            ----------------------------
            _Tracekr - tracker_instance
        """
        return cv2.TrackerMIL_create()
    
    def init_kcf_tracker(self) -> _Tracker:
        """
            Initialize KCF (Kernelized Correlation Filter) tracker instance
            Returns: 
            ----------------------------
            _Tracekr - tracker_instance
        """
        return cv2.TrackerKCF_create()
    
    def init_tld_tracker(self) -> _Tracker:
        """
            Initialize TLD (Tracking, learning and detection) tracker instance
            Returns: 
            ----------------------------
            _Tracekr - tracker_instance
        """
        return cv2.TrackerTLD_create()

    def init_medianflow_tracker(self) -> _Tracker:
        """
            Initialize MedianFlow tracker instance
            Returns: 
            ----------------------------
            _Tracekr - tracker_instance
        """
        return cv2.TrackerMedianFLow_create()
    
    def init_mosse_tracker(self) -> _Tracker:
        """
            Initialize MOSSE (Minimum Output Sum of Squared Error) tracker instance
            Returns: 
            ----------------------------
            _Tracekr - tracker_instance
        """
        return cv2.TrackerMOSSE_create()
    
    def init_csrt_tracker(self) -> _Tracker:
        """
            Initialize CSRT tracker instance
            Returns: 
            ----------------------------
            _Tracekr - tracker_instance
        """
        return cv2.TrackerCSRT_create()
