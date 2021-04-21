import logging
import time
from functools import lru_cache
from typing import Dict, List, Tuple, Union
import cv2
import numpy as np 
from loader import Loader
from tracker import Tracker
from utils import get_box_dimensions, post_process
from config import Config


class Detector: 

    logging.getLogger(__name__)

    def __init__(self, config: Config):
        if not isinstance(config, (Config, Dict)):
            raise ValueError(
                  f"Expected tracker type to be {Config.__class__.__name__}, but got {type(config)} instead")
        self.config = config
    
    def prepare_detector(self) -> None: 
        """
            This function prepares detector to be able\n
            to fully use given YOLO model in camera\n
            Returns:
            -------------------------
            Function returns nothing, it is only responsible\n
            for initializing instance variables\n
        """
        for value in [self.config.model_weights_path, self.config.model_config_path, self.config.coco_names]:
            if value is None: 
                raise ValueError(
                      f"Encountered None value for {value}, loading yolo suspended")
        loader = Loader(model_weights_path=self.config.model_weights_path,
                        model_config_path=self.config.model_config_path,
                        coco_names=self.config.coco_names)
        if self.config.use_tracker: 
            self.tracker = Tracker.load_tracker_from_config(self.config.tracker_type)
            self.tracker.init_tracker(self.config.multi_tracker)
        if self.config.use_opencv:
            self.model, self.classes, self.output_layers, self.colors = loader.load_cv2_yolo()
        else: 
            self.session, self.classes, self.colors = loader.load_onnx_yolo()
        if self.config.inference_engine: 
            self.set_target()
            self.set_backend()
    
    def set_target(self) -> None:
        """
            Set model target to use MYRIAD processors
            present in Movidius Neural Compute Stick
        """
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    
    def set_backend(self) -> None:
        """
            Set opencv backend to use Inference Engine compatible with Intel NCS
        """
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    
    @lru_cache(maxsize=None)
    def detect(self,
               input_stream: Union[int, str]
               ) -> None:
        """
            Detect function responsible for working with camera\n
            given an input stream which can be 0 if webcamera is used\n
            or string path for video file\n
            Arguments: 
            ----------------------------------
            input_stream: Union[int, str] - int if webcamera is used or path to video file
            Returns: 
            ----------------------------------
            None - function visualize preprocessed output frame on window.
        """
        camera = cv2.VideoCapture(input_stream)
        font = cv2.FONT_HERSHEY_PLAIN
        scalefactor: Union[int, float] = self.config.scalefactor
        img_size: tuple = tuple(self.config.size)
        mean: List[int] = self.config.mean
        swapRB: bool = self.config.swapRB
        score_threshold: float = self.config.score_threshold
        nms_threshold: float = self.config.nms_threshold
        timestamp_init_frames: int = self.config.timestamp_init_frames
        conf_threshold: float = self.config.conf_threshold
        write_perf_info: bool = self.config.write_perf_information
        skip_frames: int = self.config.skip_frames
        max_distance: int = self.config.max_distance
        is_on_rails_distance: int = self.config.is_on_rails_distance
        line: np.ndarray = np.array(self.config.line)

        if not camera.isOpened(): 
            logging.error(
                "Detection suspended, camera is not opened, check your configuration"
            )
            return
            
        frame_counter = 0 
        while True: 
            _, frame = camera.read()
            if frame_counter % skip_frames != 0: 
                frame_counter += 1 
                continue
            before = time.time()
            frame_shape: Tuple[int] = frame.shape
            blob = cv2.dnn.blobFromImage(image=frame,
                                         scalefactor=scalefactor,
                                         size=img_size,
                                         mean=mean,
                                         swapRB=swapRB)
            if self.config.use_opencv:
                self.model.setInput(blob)
                outputs = self.model.forward(self.output_layers)
                outputs = np.concatenate(outputs, axis=0)
            else: 
                outputs = self.session.run(["classes", "boxes"], {"images": blob})
                outputs = np.concatenate([outputs[1], outputs[0]], axis=1)
            
            matches = outputs[np.where(np.max(outputs[:, 4:], axis=1) > conf_threshold)]

            boxes, scores, class_ids = get_box_dimensions(matches,  
                                                          frame_shape[0],
                                                          frame_shape[1],
                                                          self.config.use_opencv)
            
            indexes = cv2.dnn.NMSBoxes(boxes,
                                       scores,
                                       score_threshold,
                                       nms_threshold)

            boxes, class_ids = post_process(boxes, class_ids, indexes)

            if timestamp_init_frames:
                if boxes is not None: 
                    self.tracker.feed_init_frame(multi_tracker=self.config.multi_tracker,
                                                 frame=frame,
                                                 bbox=boxes)
            
            self.detect_objects_near_line(boxes,
                                          class_ids,
                                          line,
                                          max_distance,
                                          is_on_rails_distance)
            
            for i in range(len(boxes)):
                if i in indexes: 
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    color: np.ndarray = self.colors[i]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color)
                    cv2.putText(frame, label, (x, y - 5), font, 2, color)
            
            cv2.imshow("Frame", frame)
            frame_counter += 1 
            if write_perf_info:
                after = time.time()
                logging.info(f"FPS: {1/(after-before)}")
            
            if cv2.waitKey(1) == ord("q"):
                break
        
        cv2.destroyAllWindows()
        camera.release()
    
    def detect_objects_near_line(self,
                                 boxes: List[List[int]],
                                 class_ids: List[int],
                                 line: np.ndarray,
                                 max_distance: int,
                                 is_on_rails_distance:int):
        """
            Function responsible for detecting objects that are coming
            to rails, defining from which direction they are coming and also defining
            if one of them is on rails at the moment
            Arguments: 
            ------------------------------------
            boxes - List[List[int]] - List of boxes given either by the YOLO or Detector model
            class_ids - List[int] - class indexes for YOLO classes
            line - np.ndarray - Line defining rails existance
            max_distance - int - maximum distance between object centroid and area where
            we need to take care of object coming near rails
            is_on_rails_distance - int - distance between object centroid and rails where 
            we detect object on rails at the moment 
            Returns:
            ----------------------------------------
            None
        """
        mean_y = np.mean([line[1], line[3]])
        mean_x = np.mean([line[0], line[2]])
        for i, box in enumerate(boxes):
            centroid = np.array([box[0]+box[2]/2, box[1]+box[3]/2])
            distance_y = centroid[1] - mean_y
            distance_x = centroid[0] - mean_x 
            if centroid[1] > mean_y or centroid[0] < mean_x: 
                if distance_y < max_distance or distance_x < max_distance: 
                    if distance_y < is_on_rails_distance or distance_x < is_on_rails_distance:
                        logging.info(
                            f"Object {str(self.classes[class_ids[i]])} is on rails"
                        )
                    else:
                        logging.info(
                            f"Object {str(self.classes[class_ids[i]])} coming from bottom left"
                        )
            elif centroid[1] < mean_y and centroid[0] < mean_x: 
                if distance_y < max_distance or distance_x < max_distance: 
                    if distance_y < is_on_rails_distance or distance_x < is_on_rails_distance:
                        logging.info(
                            f"Object {str(self.classes[class_ids[i]])} is on rails"
                        )
                    else: 
                        logging.info(
                            f"Object {str(self.classes[class_ids[i]])} coming from top left"
                        )
            elif centroid[1] > mean_y and centroid[0] > mean_x:
                if distance_y < max_distance or distance_x < max_distance: 
                    if distance_y < is_on_rails_distance or distance_x < is_on_rails_distance:
                        logging.info(
                            f"Object {str(self.classes[class_ids[i]])} is on rails"
                        )
                    else: 
                        logging.info(
                            f"Object {str(self.classes[class_ids[i]])} coming from bottom right"
                        )
            elif centroid[1] < mean_y and centroid[0] > mean_x: 
                if distance_y < max_distance or distance_x < max_distance: 
                    if distance_y < is_on_rails_distance or distance_x < is_on_rails_distance:
                        logging.info(
                            f"Object {str(self.classes[class_ids[i]])} is on rails"
                        )
                    else: 
                        logging.info(
                            f"Object {str(self.classes[class_ids[i]])} coming from top right"
                        )
