import pandas as pd
import argparse
from src import Vocabulary, training_dataset, testing_dataset, Collate_fn
from src import model as Model
from src import BatchNormalisation, InstanceNormalisation, BatchInstanceNormalisation, LayerNormalisation, GroupNormalisation
import torch
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--norm_type", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

norm_to_class = {"Baseline": [torch.nn.LazyBatchNorm2d], "NN": [], "BN": [BatchNormalisation], "IN": [InstanceNormalisation], "BIN": [BatchInstanceNormalisation], "LN": [LayerNormalisation], "GN": [GroupNormalisation]}

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224, 224))
])
val_dataset = testing_dataset(args.data_path, val_transform)

BATCH_SIZE=32

from torch.utils.data import DataLoader
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn = lambda batch: Collate_fn(batch, train=False), num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(*norm_to_class[args.norm_type]).to(device)
with torch.no_grad():
    model(torch.rand(1, 3, 224, 224))
model.load_state_dict(torch.load(args.model_path, map_location=device)["model"])

@torch.inference_mode()
def inference_loop(loader, model, device):
    model.eval()
    all_logits = []
    for batch in loader:
        X = batch["img"].to(device, non_blocking=True)
        print(X.shape)
        logits = model(X)
        all_logits.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    return  all_logits

import time
start_time=time.time()
y_pred = inference_loop(val_loader, model, device)
df = {
    "image_name": val_dataset.img_name,
    "predicted_label": y_pred
}
df = pd.DataFrame(df)
df.to_csv(f"{args.save_dir}/predictions.csv")

end_time=time.time()
print(f"Time taken for training: {end_time-start_time}")
