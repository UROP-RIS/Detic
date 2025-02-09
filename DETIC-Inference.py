# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
import numpy as np
import os
import cv2
import tqdm
import sys
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, "third_party/CenterNet2/")
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# Constant
INPUT_ROOT = "/data/datasets/zguobd/reasoning_caption_generator/data/refcoco/train2014"
OUTPUT_ROOT = (
    "/data/datasets/zguobd/reasoning_caption_generator/data/refcoco/train2014_detic"
)


BATCH_SIZE = 16

# INPUT_ROOT = "/data/datasets/zguobd/reasoning_caption_generator/data/demo/input"
# OUTPUT_ROOT = "/data/datasets/zguobd/reasoning_caption_generator/data/demo/output"


def get_imgs_from_directory(directory):
    images_path = [file for file in Path(directory).glob("*.jpg") if file.is_file()]
    assert len(images_path) > 0, "No images found in {}".format(directory)
    return images_path


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom"],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def setup_model(args):
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)
    predictor = demo.predictor
    model = predictor.model
    model.eval()
    aug = predictor.aug
    thing_classes = demo.metadata.thing_classes
    return model, predictor, aug, thing_classes


class ImageDataset(Dataset):

    def __init__(self, images_path, aug):
        self.images_path = images_path
        self.aug = aug

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path = self.images_path[index].as_posix()
        img_name = self.images_path[index].stem
        raw_img = read_image(img_path, format="RGB")
        height, width = raw_img.shape[:2]
        img = self.aug.get_transform(raw_img).apply_image(raw_img)
        img = img.transpose(2, 0, 1).astype("float32")
        img_tensor = torch.as_tensor(img)
        return raw_img, img_tensor, img_name, height, width


def collate_fn(batch):
    batch_inputs = []
    img_names = []
    raw_imgs = []
    for raw_img, img_tensor, img_name, height, width in batch:
        batch_inputs.append({"image": img_tensor, "height": height, "width": width})
        img_names.append(img_name)
        raw_imgs.append(raw_img)
    return batch_inputs, img_names, raw_imgs


def save_visualization(
    img, img_name, boxes, scores, classes, output_dir, thing_classes
):

    ## Draw the bounding box on the image
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    img_copy = img.copy()
    ## Convert the image to BGR format
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 1)

        label = f"{thing_classes[cls]}: {score:.2f}"

        cv2.putText(
            img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    path = os.path.join(output_dir, img_name + "_detect" + ".jpg")
    cv2.imwrite(path, img_copy)


def save(boxes, scores, classes, thing_classes, img_name, output_dir):
    ## Save the bounding boxes, scores, and classes to npy files
    classes_label = list(map(lambda x: thing_classes[x], classes))
    path = os.path.join(output_dir, img_name + "_detect" + ".npz")
    data = {
        "boxes": boxes,
        "scores": scores,
        "classes": classes_label,
    }
    np.savez(path, **data)


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    model, predictor, aug, thing_classes = setup_model(args)
    images = get_imgs_from_directory(INPUT_ROOT)

    dataset = ImageDataset(images, aug)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
    )

    for i, (batch_inputs, batch_names, raw_imgs) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            predictions = model(batch_inputs)
            for raw_img, prediction, img_name in zip(
                raw_imgs, predictions, batch_names
            ):
                instances = prediction["instances"].to("cpu")
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                classes = instances.pred_classes.numpy()
                # save_visualization(
                #     raw_img,
                #     img_name,
                #     boxes,
                #     scores,
                #     classes,
                #     OUTPUT_ROOT,
                #     thing_classes,
                # )
                save(boxes, scores, classes, thing_classes, img_name, OUTPUT_ROOT)


# if __name__ == "__main__":
#     args = get_parser().parse_args()
#     setup_logger(name="fvcore")
#     logger = setup_logger()
#     logger.info("Arguments: " + str(args))
#     model, predictor, aug = setup_model(args)
#     # prepare image
#     path = args.input[0]
#     raw_img = read_image(path, format="RGB")

#     height, width = raw_img.shape[:2]
#     img = aug.get_transform(raw_img).apply_image(raw_img)
#     img = img.transpose(2, 0, 1).astype("float32")
#     img_tensor = torch.as_tensor(img).to(model.device)

#     inputs = {"image": img_tensor, "height": height, "width": width}
#     with torch.no_grad():
#         predictions = model([inputs])[0]

#     ## Draw the bounding box on the image
#     instances = predictions["instances"].to("cpu")
#     boxes = instances.pred_boxes.tensor.numpy()
#     scores = instances.scores.cpu().numpy()
#     classes = instances.pred_classes.cpu().numpy()

#     COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
#     img_copy = raw_img.copy()
#     for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
#         x1, y1, x2, y2 = map(int, box)
#         color = COLORS[i % len(COLORS)]
#         cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

#         label = f"{cls}: {score:.2f}"

#         cv2.putText(
#             img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
#         )

#     cv2.imwrite("./output.jpg", img_copy)
