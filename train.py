import argparse
from src import Vocabulary, training_dataset, Collate_fn
from src import model as Model
from src import BatchNormalisation, InstanceNormalisation, BatchInstanceNormalisation, LayerNormalisation, GroupNormalisation
import torch
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str)
parser.add_argument("--val_path", type=str)
parser.add_argument("--norm_type", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

norm_to_class = {"Baseline": [torch.nn.LazyBatchNorm2d], "NN": [], "BN": [BatchNormalisation], "IN": [InstanceNormalisation], "BIN": [BatchInstanceNormalisation], "LN": [LayerNormalisation], "GN": [GroupNormalisation]}

train_transform = transforms.Compose([
    transforms.RandomCrop((224, 224), pad_if_needed=True),
    transforms.RandomHorizontalFlip(p=0.5)
])
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224, 224))
])
val_dataset = training_dataset(args.val_path, val_transform)
train_dataset = training_dataset(args.train_path, train_transform)

BATCH_SIZE=256
EPOCHS=75

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn = lambda batch: Collate_fn(batch), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn = lambda batch: Collate_fn(batch), num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(*norm_to_class[args.norm_type]).to(device)
temp_X = next(iter(train_loader))["img"]
model(temp_X.to(device))
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

print(f"Device used: {device}")

def training_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    count=0
    all_logits, all_labels=[], []
    for batch in loader:
        X, y = batch["img"].to(device, non_blocking=True), batch["label_idx"].to(device, non_blocking=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count+=loss.item()
        all_logits.append(torch.argmax(logits.detach().cpu(), dim=-1))
        all_labels.append(y.detach().cpu())
    history={
        "y_true": torch.concat(all_labels),
        "y_pred": torch.concat(all_logits),
        "loss": count/len(loader)
    }
    return history

@torch.inference_mode()
def inference_loop(loader, model, loss_fn, device):
    model.eval()
    count=0
    all_logits, all_labels=[], []
    for batch in loader:
        X, y = batch["img"].to(device, non_blocking=True), batch["label_idx"].to(device, non_blocking=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        count+=loss.item()
        all_logits.append(torch.argmax(logits.detach().cpu(), dim=-1))
        all_labels.append(y.detach().cpu())
    history={
        "y_true": torch.concat(all_labels),
        "y_pred": torch.concat(all_logits),
        "loss": count/len(loader)
    }
    return history

import time
start_time=time.time()
for epoch in range(EPOCHS):
    training_history = training_epoch(train_loader, model, optimizer, loss_fn, device)
    training_loss=training_history["loss"]
    validation_history = inference_loop(val_loader, model, loss_fn, device)
    validation_loss=validation_history["loss"]
    torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "training_history": training_history,
                "validation_history": validation_history
            }, f"{args.save_dir}/{args.norm_type}/epoch_{epoch}.pt"
        )
    print(f"EPOCH: {epoch+1} |  Training loss: {training_loss} | Validation loss: {validation_loss} | Ended at {time.time()-start_time}")
torch.save({
    "model": model.state_dict()
    }, f"{args.save_dir}/{args.norm_type}.pt"
)
end_time=time.time()
print(f"Time taken for training: {end_time-start_time}")
