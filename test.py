import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from FashionMnist import FashionCNN
from FashionMnist import test_loader
# Hyper-parameters 
num_epochs = 5
batch_size = 100
learning_rate = 0.001

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Đường dẫn đến file chứa model_state_dict đã lưu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model_state_dict.pt"
model = FashionCNN().to(device)
# Nạp model_state_dict vào mô hình mới
#model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


with torch.no_grad():
    n_correct = 0
    n_samples = 0
 
    for i, (images, labels) in enumerate (tqdm(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

