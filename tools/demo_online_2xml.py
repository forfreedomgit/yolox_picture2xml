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

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from lxml.etree import Element, SubElement, tostring

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


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def create_xml(list_xml, list_images):

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'Images'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(list_images[3])
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(list_images[1])
    node_height = SubElement(node_size, 'height')
    node_height.text = str(list_images[0])
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(list_images[2])

    if len(list_xml) >= 1:        # 循环写入box
        for list_ in list_xml:
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            # if str(list_[4]) == "person":                # 根据条件筛选需要标注的标签,例如这里只标记person这类，不符合则直接跳过
            #     node_name.text = str(list_[4])
            # else:
            #     continue
            node_name.text = str(list_[4])
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(list_[0])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(list_[1])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(list_[2])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(list_[3])

        # node_object = SubElement(node_root, 'object')
        # node_name = SubElement(node_object, 'name')
        # # if str(list_[4]) == "person":                # 根据条件筛选需要标注的标签,例如这里只标记person这类，不符合则直接跳过
        # #     node_name.text = str(list_[4])
        # # else:
        # #     continue
        # node_name.text = list_xml[4]
        # node_difficult = SubElement(node_object, 'difficult')
        # node_difficult.text = '0'
        # node_bndbox = SubElement(node_object, 'bndbox')
        # node_xmin = SubElement(node_bndbox, 'xmin')
        # node_xmin.text = str(list_xml[0])
        # node_ymin = SubElement(node_bndbox, 'ymin')
        # node_ymin.text = str(list_xml[1])
        # node_xmax = SubElement(node_bndbox, 'xmax')
        # node_xmax.text = str(list_xml[2])
        # node_ymax = SubElement(node_bndbox, 'ymax')
        # node_ymax.text = str(list_xml[3])

    xml = tostring(node_root, pretty_print=True)   # 格式化显示，该换行的换行

    file_name = list_images[3].split(".")[0]
    filename = 'D:/AI/YOLOX-main/datasets/WSW/wsw_yuantu/yauntu/00previous_datasets/picture_xml_20220427/xml/' + '{}.xml'.format(file_name)

    f = open(filename, "wb")
    f.write(xml)
    f.close()


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

        img_name = os.path.basename(img)
        img = cv2.imread(img)

        height, width, depth = img.shape[:3]

        list_images_ = (height, width, depth, img_name)

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
        return outputs, list_images_


def image_demo(predictor, vis_folder, path, current_time, save_result, test_size_=640):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()

    # print(VOC_CLASSES) = ('WBC', 'SEC', 'GNB', 'GPC', 'GPB', 'GNC', 'AFB', 'GPLQJ')

    # print(VOC_CLASSES[0])

    for image_name in files:
        outputs_, list_images__ = predictor.inference(image_name)
        ratio = min(test_size_ / list_images__[0], test_size_ / list_images__[1])
        # print(len(outputs_[0]), type(outputs_[0]))  # 1 <class 'list'>
        # print(len(outputs_), len(outputs_[0]), outputs_[0])
        # 1 (1743, 2501, 3, 'Augly_cutmix_6200.jpg') C:/Users/chuxi/Desktop/test/Augly_cutmix_6200.jpg
        # 1 <class 'list'> outputs_
        # list_xml_ = [outputs_[i][0], outputs_[i][1], outputs_[i][2], outputs_[i][3], outputs_[i][6]]
        # list_xml_ = [outputs_[:, 0][i], outputs_[:, 1][i], outputs_[:, 2][i], outputs_[:, 3][i], outputs_[:, 6][i]]
        #
        # create_xml(list_xml_, list_images__)
        # print(len(outputs_[i]))  # 105
        # if outputs_[0]:
        #     print('feikong')
        # else:
        #     print('kong')
        try:

            list_xml__ = []
            for list__ in outputs_[0]:
            # if list__:

            # print(list__)

            #     print(len(list_[:].cpu().numpy()))
                list_xml_ = [int(list__[0].cpu().round().item() / ratio),
                         int(list__[1].cpu().round().item() / ratio),
                         int(list__[2].cpu().round().item() / ratio),
                         int(list__[3].cpu().round().item() / ratio),
                         str(VOC_CLASSES[int(list__[6].cpu().item())])]
                list_xml__.append(list_xml_)
            # print(list_xml__)
            # print(str(VOC_CLASSES[int(list__[6].cpu().item())]))
            # print(list_xml_)
                create_xml(list_xml__, list_images__)
        except:
            pass
        continue
        # if outputs_[0]:
        #     list_xml__ = []
        #
        #     for list__ in outputs_[0]:
        #         # if list__:
        #
        #         # print(list__)
        #
        #         #     print(len(list_[:].cpu().numpy()))
        #         list_xml_ = [int(list__[0].cpu().round().item() / ratio),
        #                      int(list__[1].cpu().round().item() / ratio),
        #                      int(list__[2].cpu().round().item() / ratio),
        #                      int(list__[3].cpu().round().item() / ratio),
        #                      str(VOC_CLASSES[int(list__[6].cpu().item())])]
        #         list_xml__.append(list_xml_)
        #         # print(list_xml__)
        #         # print(str(VOC_CLASSES[int(list__[6].cpu().item())]))
        #         # print(list_xml_)
        #         create_xml(list_xml__, list_images__)
        #
        # else:
        #     continue


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_PLAIN
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # save_folder = os.path.join(
    #     vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # )
    # os.makedirs(save_folder, exist_ok=True)
    # if args.demo == "video":
    #     save_path = os.path.join(save_folder, args.path.split("/")[-1])
    # else:
    #     save_path = os.path.join(save_folder, "camera.mp4")
    # logger.info(f"video save_path is {save_path}")
    # vid_writer = cv2.VideoWriter(
    #     save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    # )
    while True:
        ret_val, frame = cap.read()

        outputs, img_info = predictor.inference(frame)
        result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
        # if args.save_result:
        #     vid_writer.write(result_frame)
        # cv2.putText(result_frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)

        cv2.namedWindow('video', 0)
        cv2.resizeWindow('video', 1280, 800)
        cv2.imshow("video", result_frame)

        ch = cv2.waitKey(1)
        # cv2.destroyAllWindows()

        if ch == ord("q"):
            break

        # cap.release()
        # cv2.destroyAllWindows()


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
