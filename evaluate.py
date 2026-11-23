import torch
from sklearn import metrics
import pandas as pd

models_dir = ["Baseline", "NN", "BN", "IN", "BIN", "LN", "GN", "experiment/BN", "experiment/GN"]
dir_to_model = {"Baseline": "Baseline", "NN": "No Normalisation", "BN": "Batch Normalisation", "IN": "Instance Normalisation", "BIN": "Batch Instance Normalisation", "LN": "Layer Normalisation", "GN": "Group Normalisation", "experiment/BN": "BN_S", "experiment/GN": "GN_S"}

result = {"Model Varient": ["Baseline", "No Normalisation", "Batch Normalisation", "Instance Normalisation", "Batch Instance Normalisation", "Layer Normalisation", "Group Normalisation", "Batch Normalisation with Batch Size 8", "Group Normalisation with Batch Size 8"],
          "Training Accuracy": [],
          "Training F1 score(micro)": [],
          "Training F1 score(macro)": [],
          "Validation Accuracy": [],
          "Validation F1 score(micro)": [],
          "Validation F1 score(macro)": []
          }
for model in models_dir:
    history = torch.load(f"checkpoint/{model}_result.pt", map_location=torch.device('cpu'))
    print(f"{history["training_history"]}")
    result["Training Accuracy"].append(metrics.accuracy_score(history["training_history"]["y_true"], history["training_history"]["y_pred"]))
    result["Training F1 score(micro)"].append(metrics.f1_score(history["training_history"]["y_true"], history["training_history"]["y_pred"], average='micro'))
    result["Training F1 score(macro)"].append(metrics.f1_score(history["training_history"]["y_true"], history["training_history"]["y_pred"], average='macro'))
    result["Validation Accuracy"].append(metrics.accuracy_score(history["validation_history"]["y_true"], history["validation_history"]["y_pred"]))
    result["Validation F1 score(micro)"].append(metrics.f1_score(history["validation_history"]["y_true"], history["validation_history"]["y_pred"], average='micro'))
    result["Validation F1 score(macro)"].append(metrics.f1_score(history["validation_history"]["y_true"], history["validation_history"]["y_pred"], average='macro'))
    print(f"Model Varient: {dir_to_model[model]} | Training Accuracy: {result["Training Accuracy"][-1]} | Training F1 score(micro): {result["Training F1 score(micro)"][-1]} | Training F1 score(macro): {result["Training F1 score(macro)"][-1]} | Validation Accuracy: {result["Validation Accuracy"][-1]} | Validation F1 score(micro): {result["Validation F1 score(micro)"][-1]} | Validation F1 score(macro): {result["Validation F1 score(macro)"][-1]}")
print(pd.DataFrame(result).to_latex())

EPOCHS=75

import matplotlib.pyplot as plt
import numpy as np

for model in models_dir:
    try:
        history = torch.load(f"checkpoint/{model}_loss.pt", map_location=torch.device('cpu'))
        training_loss = history["training_loss"]
        validation_loss = history["validation_loss"]
        fig, ax = plt.subplots()
        ax.plot(np.arange(EPOCHS), training_loss, color='r', label='Training Loss')
        ax.plot(np.arange(EPOCHS), validation_loss, color='y', label='Validation Loss')
        ax.set_title(f"Model with {dir_to_model[model]}")
        ax.set_xlabel("Number Of Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.savefig(f"results/{model}_loss.png")
        plt.show()
    except:
        continue
print("End Of Script")
