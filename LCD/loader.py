from dataclasses import dataclass
from typing import Dict, List, Tuple, TypeVar, Union
import onnxruntime
import cv2
import numpy as np 
from config import Config

opencv_model = TypeVar("opencv_model")

@dataclass
class Loader: 
    model_weights_path: str
    model_config_path: str
    coco_names: str

    @classmethod
    def load_from_config(cls, 
                         config: Union[Dict, Config], 
                         **kwargs):
        """
            Class method responsible for more user friendly
            config loading.
            Arguments:
            ------------------------------
            config - Dict - loaded config dict with predefined
            variables for loading YOLO model 
            Returns: 
            ------------------------------
            cls - Loader class object with defined instance variables from config
        """
        if isinstance(config, Dict):
            for item, value in config.items():
                if value is None: 
                    raise ValueError(
                        f"Encountered None value for item {item}, loading yolo suspended"
                        )
            return cls(**config)
        if isinstance(config, Config):
            for value in [config.model_weights_path, config.model_config_path, config.coco_names]:
                if value is None:
                    raise ValueError(
                        f"Encountered None value for value {value}, loading yolo suspended"
                        )
            return cls(config.model_weights_path,
                       config.model_config_path,
                       config.coco_names)

    def load_cv2_yolo(self) -> Tuple[opencv_model,
                                     List[str],
                                     List[str],
                                     np.ndarray]:
        """
             Function responsible for loading YOLO model from OpenCV
             which is more efficient and faster way than reading YOLO
             from Darknet for example (recommended to use this function
             when you dealing with no GPU problems)
             Returns: 
             -------------------------------
             opencv_model - loaded_opencv_model ready to be used in inference\n
             classes - List[str] - COCO classes_names\n
             output_layers - List[str] - List of model layers which will be used
             to propagate input in inference mode\n
             colors: np.ndarray - unique color array for each class
        """
        model = cv2.dnn.readNet(self.model_weights_path,
                                self.model_config_path)
        with open(self.coco_names, 'r') as f: 
            classes = list(map(lambda x: x.strip(), f.readlines()))
        layers_name = model.getLayerNames()
        output_layers = list(map(lambda x: layers_name[x[0] - 1],
                                 model.getUnconnectedOutLayers()))
        colors: np.ndarray = np.random.uniform(0, 255, size=(len(classes), 3))

        return model, classes, output_layers, colors
    
    def load_onnx_yolo(self) -> Tuple[onnxruntime.InferenceSession,
                                      List[str], np.ndarray]:
        """
            Load YOLOv3 compatible with ONNX ecosystem
            Returns: 
            -------------------------------
            session - onnxruntime.InferenceSession - ONNX model instance\n
            classes - List[str] - COCO classes_names\n
            colors: np.ndarray - unique color array for each class
        """
        session = onnxruntime.InferenceSession(self.model_weights_path)
        session.get_modelmeta()
        with open(self.coco_names, 'r') as f: 
            classes = list(map(lambda x: x.strip(), f.readlines()))
        colors: np.ndarray = np.random.uniform(0, 255, size=(80, 3))
        return session, classes, colors
