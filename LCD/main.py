import logging
import argparse
from detector import Detector
from config import Config

parser = argparse.ArgumentParser(description="YOLO CPU Model")

parser.add_argument("--input-stream",
                    dest="input_stream",
                    help="Input stream param, 0 stands for camera, otherwise path to video file",
                    default=0,
                    type=str)

parser.add_argument("--config-path",
                    dest="config_path",
                    help="config file with predefined parameters to run program",
                    type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    config = Config.load_config_class(args.config_path)
    logging.basicConfig(level=config.level,
                        filename=config.filename,
                        format="[%(asctime)s] [%(levelname]7s] [%(funcName)s] [%(name)12s:%(lineno)s] -- %(message)s")
    detector = Detector(config)
    detector.prepare_detector()
    detector.detect(args.input_stream)
