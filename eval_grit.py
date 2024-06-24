
import sys
sys.path.insert(0, "third_party/CenterNet2/projects/CenterNet2/")
from centernet.config import add_centernet_config


import os
import inspect
import json, tqdm
import logging
import torch
from argparse import Namespace
import detectron2.data.transforms as T
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.structures import Boxes, Instances
from pycocotools.coco import COCO
from grit.config import add_grit_config
from grit.predictor import Visualizer_GRiT

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)
filname = os.getenv("LOG_FILENAME", "region_caption.log")
logging.basicConfig(level=logging.INFO, filename=filname, filemode="w", force=True)

class GRiTInferenceEngine:
    def setup_cfg(self, args):
        cfg = get_cfg()
        if args.cpu:
            cfg.MODEL.DEVICE = "cpu"
        add_centernet_config(cfg)
        add_grit_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        
        if args.test_task:
            cfg.MODEL.TEST_TASK = args.test_task
        cfg.MODEL.BEAM_SIZE = 1
        cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
        cfg.USE_ACT_CHECKPOINT = False
        cfg.freeze()
        return cfg

    def __init__(self, args):
        # TODO: move obj outside of the class, for gradio reload
        # The size of the image may be changed by the augmentations.
        # NOTE: from demo/demo.py
        cfg = self.setup_cfg(args)
        # self.predictor = DefaultPredictor(cfg)

        # NOTE: from demo/predictor.py:VisualizationDemo
        cpu_device = torch.device("cpu")
        instance_mode = ColorMode.IMAGE

        # NOTE: from detectron2/engine/defaults.py:DefaultPredictor
        model = build_model(cfg)
        model.eval()
        logger.info(f"meta_arch: {type(model)} from {inspect.getfile(type(model))}")

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        input_format = cfg.INPUT.FORMAT
        assert input_format in ["RGB", "BGR"], input_format

        # NOTE: from detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN, only by setting `proposal_generator=None`, the model will use the proposals in inputs
        # Otherwise it will generate proposals by itself
        # NOTE: arg `detected_instances` is the detection results, which is for mask and keypoint (at least in GeneralizedRCNN). It is not the proposals.
        self.proposal_generator = model.proposal_generator

        self.cfg = cfg
        self.cpu_device = cpu_device
        self.instance_mode = instance_mode
        self.model = model
        self.aug = aug
        self.input_format = input_format

        self.aug_transform = None
        self.image = None
        self.inputs = None
        self.encoded_image_dict = None
        
        self.img_dir = args.img_dir
        self.dataset = COCO(args.anno_file)
        self.res = {}
        self.gts = {}


    @torch.no_grad()
    def _encode_image(self, input_image):
        if not isinstance(input_image, Image.Image):
            raise ValueError(
                f"input_image should be PIL.Image.Image, got {type(input_image)}"
            )
        original_image = convert_PIL_to_numpy(input_image, self.input_format)
        # return self.predictor(original_image)

        height, width = original_image.shape[:2]
        aug_transform = self.aug.get_transform(original_image)
        image = aug_transform.apply_image(original_image)
        image = torch.as_tensor(
            image.astype("float32").transpose(2, 0, 1)
        )  # HWC -> CHW
        logger.info(
            f"original_image.shape: {original_image.shape}; image.shape: {image.shape}"
        )

        inputs = {"image": image, "height": height, "width": width}
        encoded_image_dict = self.model.encode_image([inputs])
        return aug_transform, image, inputs, encoded_image_dict

    def inference_dataset(
        self, 
    ):
        results = []
        for img_id in tqdm.tqdm(self.dataset.getImgIds()):
            img_info = self.dataset.loadImgs(img_id)
            image_id, image_path = img_info[0]['id'],img_info[0]['file_name']
            input_image = Image.open(os.path.join(self.img_dir, image_path)).convert('RGB')
            ann_id = self.dataset.getAnnIds(img_id)
            anns = self.dataset.loadAnns(ann_id)
            input_boxes, gt_captions = [], []
            for ann in anns:
                x1, y1, w, h = ann["bbox"]
                x2, y2 = x1 + w, y1 + h
                cap = ann["caption"]
                input_boxes.append([x1, y1, x2, y2])
                gt_captions.append(cap)

            input_boxes = np.array(input_boxes)

            pred_captions = self.inference_text(input_image, input_boxes)
            if len(pred_captions) != len(gt_captions):
                raise ValueError(
                    f"len(pred_captions) != len(gt_captions): {len(pred_captions)} != {len(gt_captions)} at {img_id}"
                )
            for i, (gt, pred) in enumerate(zip(gt_captions, pred_captions)):
                key = f"{img_id}_region_{i}"
                self.res[key] = [{"caption": pred[0]}]
                self.gts[key] = [{"caption": gt}]

            results.append({
                "image_id": img_id,
                "gt_captions": gt_captions,
                "bboxes": input_boxes.tolist(),
                "pred_captions": pred_captions
            })
        tokenizer = PTBTokenizer()
        res, gts = tokenizer.tokenize(self.res), tokenizer.tokenize(self.gts)
        # Evaluate results for each metric.
        for metric in (Cider(), Meteor(), Bleu(),Rouge(), Spice()):
            kwargs = {"verbose": 0} if isinstance(metric, Bleu) else {}
            score, _ = metric.compute_score(gts, res, **kwargs)
            print(metric.method(), score)
        json.dump(results, open("results.json", 'w'))
        


    def inference_text(self, input_image, input_boxes):
        """_summary_

        Args:
            input_image (_type_): PIL image
            input_boxes (_type_): Nx4 np array or list of list. The coordinates are in the original image space.

        Returns:
            _type_: return list of list of str
        """
        predictions = self._inference_model(input_image, input_boxes)
        pred_captions = predictions["instances"].pred_object_descriptions.data

        if isinstance(pred_captions[0], str):
            pred_captions = [[i] for i in pred_captions]
        return pred_captions

    def inference_visualization(self, input_image, input_boxes):
        """_summary_

        Args:
            input_image (_type_): PIL image
            input_boxes (_type_): Nx4 np array or list of list. The coordinates are in the original image space.

        Returns:
            _type_: np.array, The visualization of the inference result. By GRiT adapted Detectron2 visualizer
        """
        predictions = self._inference_model(input_image, input_boxes)
        output_image = self._post_process_output(input_image, predictions)

        return output_image

    # NOTE: GRiT only supports batch size 1
    # NOTE: otherwise, OOM error
    @torch.no_grad()
    def _inference_model(
        self,
        input_image: Image.Image,
        input_boxes=None,
        input_points=None,
    ):
        aug_transform, image, inputs, encoded_image_dict = self._encode_image(
            input_image
        )

        if input_points is None and input_boxes is None:
            raise ValueError("input_points or input_boxes should not be None.")
        elif input_points is not None and input_boxes is not None:
            raise ValueError(
                "input_points and input_boxes should not be both not None."
            )
        elif input_points is not None and input_boxes is None:
            input_points = np.asarray(input_points)
            input_boxes = np.concatenate([input_points, input_points], axis=-1)

        # NOTE: from detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN, only by setting `proposal_generator=None`, the model will use the proposals in inputs
        self.model.proposal_generator = None
        # NOTE: from fvcore/transforms/transform.py:apply_box, which takes in Nx4 np array.
        # It call `detectron2/data/transforms/transform.py:apply_coords` which casts the dtype to fp32.
        input_boxes = aug_transform.apply_box(input_boxes)
        input_height, input_width = image.shape[-2:]
        num_boxes = len(input_boxes)
        proposals = Instances(
            (input_height, input_width),
            proposal_boxes=Boxes(input_boxes),
            scores=torch.ones(num_boxes),
            objectness_logits=torch.ones(num_boxes),
        )
        inputs.update(dict(proposals=proposals))

        # NOTE: The **batch system** of Detectron2 is to **use List**
        predictions = self.model.inference(
            [inputs],
            encoded_image_dict=encoded_image_dict,
            replace_pred_boxes_with_gt_proposals=True,
        )[
            0
        ]  # Assign proposals

        return predictions

    def _post_process_output(self, image, predictions):
        visualizer = Visualizer_GRiT(image, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return vis_output.get_image()


args = Namespace()
# https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
#args.config_file = "configs/GRiT_B_DenseCap.yaml"
args.config_file = "configs/GRiT_B_DenseCap_ObjectDet.yaml"
args.anno_file = "/path/to/test.json" 
args.img_dir = "path/to/images/"
args.opts = [
    "MODEL.WEIGHTS",
    "models/grit_b_densecap_objectdet.pth",
]
args.test_task = "DenseCap"
args.cpu = False


infer_engine = GRiTInferenceEngine(args)
infer_engine.inference_dataset()
