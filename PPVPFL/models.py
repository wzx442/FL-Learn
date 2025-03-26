import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
from typing import Optional

class CNN_CIFAR10(nn.Module): # d = 586,250
    """CNN model specifically designed for CIFAR-10 dataset"""
    def __init__(self, num_classes: int):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN_MNIST(nn.Module): # d = 356,298
    """CNN model specifically designed for MNIST dataset"""
    def __init__(self, num_classes: int):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLP_CIFAR10(nn.Module): # d = 3,805,450 
    """MLP model specifically designed for CIFAR-10 dataset"""
    def __init__(self, num_classes: int):
        super(MLP_CIFAR10, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class MLP_MNIST(nn.Module): # d = 567,434 
    """MLP model specifically designed for MNIST dataset"""
    def __init__(self, num_classes: int):
        super(MLP_MNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_model(model_name: str, num_classes: int, dataset: str = 'cifar10', device: Optional[torch.device] = None) -> nn.Module:
    """Load and initialize a specified model.
    
    Args:
        model_name: Name of the model to load ('cnn', 'mlp', or 'resnet18')
        num_classes: Number of output classes
        dataset: Dataset to use ('cifar10' or 'mnist')
        device: Device to load the model on (default: cuda if available, else cpu)
        
    Returns:
        nn.Module: Initialized model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = dataset.lower()
    model_name = model_name.lower()
    
    # Initialize model based on name and dataset
    if model_name == 'cnn':
        if dataset == 'cifar10':
            model = CNN_CIFAR10(num_classes)
            print("CNN_CIFAR10 model initialized")
        elif dataset == 'mnist':
            model = CNN_MNIST(num_classes)
            print("CNN_MNIST model initialized")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
    elif model_name == 'mlp':
        if dataset == 'cifar10':
            model = MLP_CIFAR10(num_classes)
            print("MLP_CIFAR10 model initialized")
        elif dataset == 'mnist':
            model = MLP_MNIST(num_classes)
            print("MLP_MNIST model initialized")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
    elif model_name == 'resnet18':
        # Load ResNet18 with default weights
        model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first conv layer based on dataset
        if dataset == 'cifar10':
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # d = 3,179,082
            print("ResNet18_CIFAR10 model initialized")
        elif dataset == 'mnist':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # d = 3,177,930
            print("ResNet18_MNIST model initialized")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Modify final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Initialize weights using Kaiming initialization 使用Kaiming初始化权重
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    # Move model to device 将模型移动到设备
    model = model.to(device)
    
    # Verify no NaN values in parameters 验证参数中没有NaN值
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Warning: NaN detected in {name}")
            param.data.zero_()
    
    return model 