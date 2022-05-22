import os
import shutil
import logging
import time
import argparse
from typing import List
from pathlib import Path

from lung_annotation_maker import __version__

import numpy as np
import torch
from torch.autograd import Variable

from generator.ex_generator import generate as ex_generator
from generator.pts_generator import generate as pts_generator
from annotator import CustomLog
from annotator import PointNetDenseCls

logger = logging.getLogger("Lung Annotation Maker Interface")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(CustomLog())
logger.addHandler(handler)

__current_model__ = "https://drive.google.com/file/d/1NTX4MdWIHZZjOfbWEkYAi4Qh8B3rp73i/view?usp=sharing"
__model__path = Path("../downloaded_model")


class ProgramArguments(object):
    def __init__(self) -> None:
        self.input_file = None


def parse_args() -> ProgramArguments:
    parser = argparse.ArgumentParser(description='Add annotation labels to 3D lung data.')
    parser.add_argument("--input_file", type=str, help="Path to the .exdata file")

    program_arguments = ProgramArguments()
    parser.parse_args(namespace=program_arguments)

    return program_arguments


def __download_model() -> None:
    import urllib.request
    __model__path.mkdir(parents=True, exist_ok=True)
    # TODO!!: fix corrupted download
    with urllib.request.urlopen(__current_model__) as response, open(str(__model__path / "model.pth"), 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


def __predict(coordinates: List) -> np.ndarray:
    point_set = np.asarray(coordinates).astype(np.float32)
    denorm_point_set = point_set.copy()

    logger.info(f"Normalising the dataset...")
    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist  # scale
    points = torch.from_numpy(point_set)

    if os.path.exists(str(__model__path / "model.pth")):
        logger.info(f"Loading the cached model...")
    else:
        logger.info(f"Downloading the model from Google Drive...")
        __download_model()
        logger.info(f"Sleeping for 60 seconds to update...")
        time.sleep(60)  # sleep for 60s to update

    model = str(__model__path / "model.pth")

    state_dict = torch.load(model,
                            map_location=torch.device("cpu"))
    classifier = PointNetDenseCls(k=state_dict['conv4.weight'].size()[0],
                                  feature_transform=True)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    points = points.transpose(1, 0).contiguous()
    point = Variable(points.view(1, points.size()[0], points.size()[1]))

    logger.info(f"Predicting the labels...")
    pred, _, _ = classifier(point)
    pred_choice = pred.data.max(2)[1]

    logger.info(f"Writing out the annotated file...")
    output = np.hstack((denorm_point_set, pred_choice.numpy().T))

    return output


def main():
    logger.info(f"Version: {__version__}")
    args = parse_args()

    if args.input_file is None:
        logger.critical("No input file given. See '--help'.")
        exit(-1)

    if os.path.exists(args.input_file):
        coordinates = pts_generator(args.input_file)
        __predict(coordinates)
        # output_file = os.path.join(output_dir, f"{file_name}.pts")
        # with open(output_file, 'w') as points_file:
        #     for coordinate in coordinates:
        #         points_file.write(f"{coordinate[0]:.2f} {coordinate[1]:.2f} {coordinate[2]:.2f}\n")
        #


if __name__ == '__main__':
    main()
