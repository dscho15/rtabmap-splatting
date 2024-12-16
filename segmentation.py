import numpy as np
import open3d as o3d
import o3d_utils as o3d_utils
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

import os
from typing import Literal
from PIL import Image
from distinctipy import get_colors
import torch
from torchvision.transforms.functional import resize, InterpolationMode

ID2LABEL = {
    "0": "wall",
    "1": "building",
    "2": "sky",
    "3": "floor",
    "4": "tree",
    "5": "ceiling",
    "6": "road",
    "7": "bed ",
    "8": "windowpane",
    "9": "grass",
    "10": "cabinet",
    "11": "sidewalk",
    "12": "person",
    "13": "earth",
    "14": "door",
    "15": "table",
    "16": "mountain",
    "17": "plant",
    "18": "curtain",
    "19": "chair",
    "20": "car",
    "21": "water",
    "22": "painting",
    "23": "sofa",
    "24": "shelf",
    "25": "house",
    "26": "sea",
    "27": "mirror",
    "28": "rug",
    "29": "field",
    "30": "armchair",
    "31": "seat",
    "32": "fence",
    "33": "desk",
    "34": "rock",
    "35": "wardrobe",
    "36": "lamp",
    "37": "bathtub",
    "38": "railing",
    "39": "cushion",
    "40": "base",
    "41": "box",
    "42": "column",
    "43": "signboard",
    "44": "chest of drawers",
    "45": "counter",
    "46": "sand",
    "47": "sink",
    "48": "skyscraper",
    "49": "fireplace",
    "50": "refrigerator",
    "51": "grandstand",
    "52": "path",
    "53": "stairs",
    "54": "runway",
    "55": "case",
    "56": "pool table",
    "57": "pillow",
    "58": "screen door",
    "59": "stairway",
    "60": "river",
    "61": "bridge",
    "62": "bookcase",
    "63": "blind",
    "64": "coffee table",
    "65": "toilet",
    "66": "flower",
    "67": "book",
    "68": "hill",
    "69": "bench",
    "70": "countertop",
    "71": "stove",
    "72": "palm",
    "73": "kitchen island",
    "74": "computer",
    "75": "swivel chair",
    "76": "boat",
    "77": "bar",
    "78": "arcade machine",
    "79": "hovel",
    "80": "bus",
    "81": "towel",
    "82": "light",
    "83": "truck",
    "84": "tower",
    "85": "chandelier",
    "86": "awning",
    "87": "streetlight",
    "88": "booth",
    "89": "television receiver",
    "90": "airplane",
    "91": "dirt track",
    "92": "apparel",
    "93": "pole",
    "94": "land",
    "95": "bannister",
    "96": "escalator",
    "97": "ottoman",
    "98": "bottle",
    "99": "buffet",
    "100": "poster",
    "101": "stage",
    "102": "van",
    "103": "ship",
    "104": "fountain",
    "105": "conveyer belt",
    "106": "canopy",
    "107": "washer",
    "108": "plaything",
    "109": "swimming pool",
    "110": "stool",
    "111": "barrel",
    "112": "basket",
    "113": "waterfall",
    "114": "tent",
    "115": "bag",
    "116": "minibike",
    "117": "cradle",
    "118": "oven",
    "119": "ball",
    "120": "food",
    "121": "step",
    "122": "tank",
    "123": "trade name",
    "124": "microwave",
    "125": "pot",
    "126": "animal",
    "127": "bicycle",
    "128": "lake",
    "129": "dishwasher",
    "130": "screen",
    "131": "blanket",
    "132": "sculpture",
    "133": "hood",
    "134": "sconce",
    "135": "vase",
    "136": "traffic light",
    "137": "tray",
    "138": "ashcan",
    "139": "fan",
    "140": "pier",
    "141": "crt screen",
    "142": "plate",
    "143": "monitor",
    "144": "bulletin board",
    "145": "shower",
    "146": "radiator",
    "147": "glass",
    "148": "clock",
    "149": "flag",
}


class Segmentation2D:

    def __init__(
        self,
        model_ckpt: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        cuda_device: Literal["cpu", "cuda"] = "cuda",
    ):

        self.processor = SegformerImageProcessor.from_pretrained(model_ckpt)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_ckpt)
        
        self.model.to(cuda_device) # type: ignore
        self.model.eval()
        
        self.colors = torch.tensor(get_colors(len(ID2LABEL)), dtype=torch.float32).reshape(-1, 3)

    def segment_image(self, image: np.ndarray, desired_res: tuple) -> np.ndarray:
        
        h, w = image.shape[:2]
        
        # convert to PIL image
        image = Image.fromarray(image)  # type: ignore

        inputs = self.processor(images=np.array(image), return_tensors="pt")
        
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        
        with torch.inference_mode():
            outputs = self.model(**inputs).logits.detach().cpu()
        
        outputs = outputs.softmax(1)
        
        outputs = outputs.argmax(1)
        
        outputs = self.colors[outputs][0].permute(2, 0, 1)
        
        outputs = resize(outputs, desired_res, interpolation=InterpolationMode.NEAREST)

        outputs = outputs.permute(1, 2, 0)
                
        outputs = np.clip(outputs.numpy() * 255, 0, 255).astype(np.uint8)

        return outputs # type: ignore


#     def project_segmentation_to_rgbd(
#         self, image: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray
#     ) -> np.ndarray:
#         pass


# # class Segmentation3D:

# #     def __init__(self,
# #                  model_ckpt: PathLike
# #                  depth_truncation: float = 2.5,
# #                  classes_of_interest: str[list] = classes


# #                  ):
# #         pass

# #     def segment_point_cloud_view(self, point_cloud_view: np.ndarray) -> np.ndarray:
# #         pass