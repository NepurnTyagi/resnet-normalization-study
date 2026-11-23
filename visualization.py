import torch
import numpy as np
import cv2
from torchvision import transforms
from torchvision import io
from src import BatchNormalisation
from src import model as Model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

torch.autograd.set_detect_anomaly(True)
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224, 224))
])
model = Model(BatchNormalisation)
for module in model.modules():
    if hasattr(module, 'inplace'):
        module.inplace = False
with torch.no_grad():
    model(torch.rand(1, 3, 224, 224))
model.load_state_dict(torch.load("checkpoint/BN.pt", map_location=torch.device('cpu'))["model"])
model.eval()
target_layers = [model.layer[9].layer2]
cam = GradCAM(model=model, target_layers=target_layers)
selected = [
    ("./val/n01443537/n01443537_10569.JPEG", 78, 1),
    ("./val/n01443537/n01443537_1087.JPEG", 78, 1),
    ("./val/n01443537/n01443537_11078.JPEG", 78, 1),
    ("./val/n01443537/n01443537_1118.JPEG", 78, 1),
    ("./val/n01443537/n01443537_11360.JPEG", 78, 1),
    ("./val/n01443537/n01443537_12838.JPEG", 78, 0),
    ("./val/n01443537/n01443537_13885.JPEG", 78, 0),
    ("./val/n01443537/n01443537_17565.JPEG", 78, 0),
    ("./val/n01443537/n01443537_17967.JPEG", 78, 0),
    ("./val/n01443537/n01443537_17979.JPEG", 78, 0),
]
count = 1
for img, label, category in selected:
    img =  io.read_image(img, io.ImageReadMode.RGB)/255.0
    img = val_transform(img)
    rgb_image = img.permute(1, 2, 0).numpy().astype(np.float32)
    rgb_image = np.clip(rgb_image, 0, 1)
    img = img.unsqueeze(0)
    targets = [ClassifierOutputTarget(label)]
    grayscale_cam = cam(input_tensor=img, targets=targets)
    visualization = show_cam_on_image(rgb_image, grayscale_cam[0])
    cv2.imwrite(f"results/gradcam_{label}_{category}_{count}.jpg", visualization)
    count+=1
