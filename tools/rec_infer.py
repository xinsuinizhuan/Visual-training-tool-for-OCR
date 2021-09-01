from typing import Tuple, List
import numpy as np
import torch
from torch import Tensor
from models import build_rec_model
from utils import character, load_rec_model, CTCLabelConverter
from datasets.rec_dataset import RecDataProcess
from torchvision import transforms


class RecInfer:
    def __init__(self, cfg):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_rec_model(cfg, len(character)+1)
        load_rec_model(cfg.model_path, self.model)
        self.model.to(self.device)
        self.model.eval()
        self.process = RecDataProcess(cfg.img_H)
        self.converter = CTCLabelConverter(character)
        self.transforms = transforms.ToTensor()

    def predict(self, img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        # 预处理根据训练来
        img = self.process.resize_with_specific_height(img)
        img = self.process.width_pad_img(img, 180)
        tensor = self.transforms(img)
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out: Tensor = self.model(tensor)

        txt = self.converter.decode(out.softmax(
            dim=2).detach().cpu().numpy(), False)
        return txt
