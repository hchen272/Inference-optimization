import torch
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from part1.srcnn_model import SRCNN

def load_srcnn_model(weights_path, device='cpu'):
    model = SRCNN()
    state_dict = torch.load(weights_path, map_location=device)
    # Remove 'module.' prefix if present (from DataParallel)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = ToTensor()(rgb).unsqueeze(0)
    return tensor

def postprocess_tensor(tensor):
    img = tensor.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def srcnn_upsample_frame(frame, model, device='cpu', scale=2):
    h, w = frame.shape[:2]
    bicubic_hr = cv2.resize(frame, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    input_tensor = preprocess_frame(bicubic_hr).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_frame = postprocess_tensor(output_tensor)
    return output_frame