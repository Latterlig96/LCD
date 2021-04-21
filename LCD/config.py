import pathlib
from dataclasses import dataclass
from typing import Tuple
import yaml


@dataclass
class Config(object): 

    model_weights_path: pathlib.Path
    model_config_path: pathlib.Path
    coco_names: pathlib.Path
    size: Tuple[int, int]
    scalefactor: float
    mean: Tuple[int, int, int]
    swapRB: bool
    use_opencv: bool
    score_threshold: float
    nms_threshold: float
    conf_threshold: float
    write_perf_information: bool
    skip_frames: int
    max_distance: int
    timestamp_init_frames: int
    is_on_rails_distance: int
    line: Tuple[int, int, int, int]
    filename: str
    level: str
    inference_engine: bool
    use_tracker: bool
    multi_tracker: bool
    tracker_type: str

    def __new__(cls, *args, **kwargs):
        """
            Magic Function altering class instancing
            to allow passing other config parameters 
            not specified in init constructor
        """
        init_args, additional_args = {}, {}
        for name, value in kwargs.items():
            if name in cls.__annotations__: 
                init_args[name] = value
            else: 
                additional_args[name] = value
        
        new_cls = super().__new__(cls) 
        new_cls.__init__(**init_args)

        for key, value in additional_args.items():
            setattr(new_cls, key, value)

        return new_cls
    
    @classmethod
    def load_config_class(cls, config_path: str):
        """
            Load config class with default parameters responsible to run YOLOv3
            Parameters: 
            --------------------------------
            config_path: pathlib.Path - relative path to yaml config path
            Returns: 
            --------------------------------
            Config_class: Config - Class object with filled attributes for running YOLOv3
        """
        if not isinstance(config_path, str):
            raise TypeError(
                  f"You must provide a config file with manually tuned configuration parameters, but got {config_path} instead")
        with open(config_path, 'rb') as cfg: 
            config = yaml.safe_load(cfg)
        cfg_dict = {}
        for key in config.keys():
            cfg_dict.update(**config[key])
        config_class = cls.__new__(cls, **cfg_dict)
        return config_class
