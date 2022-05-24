#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import pyautogui
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from PIL import Image
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


# path = 'assets/dog.jpg'


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=VOC_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        # if trt_file is not None:
        #     from torch2trt import TRTModule
        #
        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))
        #
        #     x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
        #     self.model(x)
        #     self.model = model_trt

    def inference(self, img):

        img = np.array(img)

        # img = cv2.imread(img)

        height, width, depth = img.shape[:3]

        list_images_ = (height, width, depth)

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])

        img, _ = self.preproc(img, None, self.test_size)

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        # img_info = {"id": 0}
        # if isinstance(img, str):
        #     img_info["file_name"] = os.path.basename(img)
        #     img = cv2.imread(img)
        # else:
        #     img_info["file_name"] = None
        #
        # height, width = img.shape[:2]
        #
        # img_info["height"] = height
        # img_info["width"] = width
        # img_info["raw_img"] = img
        #
        # ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        # img_info["ratio"] = ratio
        #
        # img, _ = self.preproc(img, None, self.test_size)
        #
        # img = torch.from_numpy(img).unsqueeze(0)
        # img = img.float()
        # if self.device == "gpu":
        #     img = img.cuda()
        #     if self.fp16:
        #         img = img.half()  # to FP16
        #
        # with torch.no_grad():
        #     t0 = time.time()
        #     outputs = self.model(img)
        #     if self.decoder is not None:
        #         outputs = self.decoder(outputs, dtype=outputs.type())
        #     outputs = postprocess(
        #         outputs, self.num_classes, self.confthre,
        #         self.nmsthre, class_agnostic=True
        #     )
        #     logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, list_images_

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        height = img_info["height"]
        width = img_info["width"]
        img = img_info["raw_img"]
        # print(img)
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        '''
        "WBC",
        "SEC",
        "GNB",
        "GPC",
        "GPB",
        "GNC",
        "AFB",
        "GPLQJ",
        '''

        box_WBC = []
        box_SEC = []
        box_GNB = []
        box_GPC = []
        box_GPB = []
        box_GNC = []
        box_AFB = []
        box_GPLQJ = []

        for i in range(len(cls)):
            if cls[i] == 0:
                box_WBC.append(cls[i])
            elif cls[i] == 1:
                box_SEC.append(cls[i])
            elif cls[i] == 2:
                box_GNB.append(cls[i])
            elif cls[i] == 3:
                box_GPC.append(cls[i])
            elif cls[i] == 4:
                box_GPB.append(cls[i])
            elif cls[i] == 5:
                box_GNC.append(cls[i])
            elif cls[i] == 6:
                box_AFB.append(cls[i])
            elif cls[i] == 7:
                box_GPLQJ.append(cls[i])
        '''
        box_WBC, box_SEC, box_GNB, box_GPC, box_GPB, box_GNC, box_AFB, box_GPLQJ
        
        WBC, SEC, GNB, GPC, GPB, GNC, AFB, GPLQJ
        '''

        print('WBC:{}, SEC:{}, GNB:{}, GPC:{}, GPB:{}, GNC:{}, AFB:{}, GPLQJ:{}'
              .format(len(box_WBC), len(box_SEC), len(box_GNB), len(box_GPC), len(box_GPB),
                      len(box_GNC), len(box_AFB), len(box_GPLQJ)))

        # bboxes_sum = {box_WBC: len(box_WBC), box_SEC: len(box_SEC), box_GNB: len(box_GNB), box_GPC: len(box_GPC),
        #               box_GPB: len(box_GPB), box_GNC: len(box_GNC), box_AFB: len(box_AFB), box_GPLQJ: len(box_GPLQJ)}
        bboxes_sum = [len(box_WBC), len(box_SEC), len(box_GNB), len(box_GPC),
                      len(box_GPB), len(box_GNC), len(box_AFB), len(box_GPLQJ)]

        number_bboxses = len(box_WBC)

        vis_res = vis(img, bboxes, scores, cls, bboxes_sum, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args, test_size_=640):

    while True:
        img = pyautogui.screenshot(region=(500, 200, 1500, 1000))
        outputs_, list_images__ = predictor.inference(img)

        ratio = min(test_size_ / list_images__[0], test_size_ / list_images__[1])

        try:

            if int(outputs_[0][0][6]) == 0:

                target_bbox = outputs_[0][0]

                # print(target_bbox/ratio)
                # (500, 200, 1500, 1000):tensor([126.4222, 420.0823, 246.3491, 781.5621,   2.2950,   2.0512,   0.0000], device='cuda:0')
                #     all                tensor([614.8744, 629.3358, 747.5789, 983.0849,   3.6143,   3.3344,   0.0000], device='cuda:0')
                # x + 500, y + 200
                target_center_x = (target_bbox[0] + target_bbox[2]).cpu().round().item() * 0.5 / ratio
                target_center_y = (target_bbox[1] + target_bbox[3]).cpu().round().item() * 0.5 / ratio
                pyautogui.moveTo(target_center_x + 500, target_center_y + 200, duration=0)
                # print(target_center_x, target_center_y)

                # pyautogui.click()
                # time.sleep(0.02)
                # pyautogui.click()
                # time.sleep(0.02)
                # pyautogui.click()
                # time.sleep(0.02)
                # pyautogui.click()
                # time.sleep(0.02)

        except:
            pass

        # ch = cv2.waitKey(1)
        #
        # if ch == ord("q"):
        #     break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, VOC_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
