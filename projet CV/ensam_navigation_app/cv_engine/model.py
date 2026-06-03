import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class MetricLearningModel(nn.Module):
    """
    Metric-learning image embedding model based on EfficientNet-B0.
    Produces a 128-dimensional L2-normalized embedding.
    """
    def __init__(self):
        super(MetricLearningModel, self).__init__()
        
        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)
        
        # Remove the original classifier head
        self.backbone.classifier = nn.Identity()
        
        # Freeze blocks 0-5 of model.features
        for param in self.backbone.features[:6].parameters():
            param.requires_grad = False
            
        # Leave blocks 6-7 (and 8) trainable for domain adaptation
        for param in self.backbone.features[6:].parameters():
            param.requires_grad = True
            
        # Add Projection Head: Linear(1280, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3) -> Linear(256, 128)
        self.projection_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        self._print_parameter_counts()

    def _print_parameter_counts(self):
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = frozen_params + trainable_params
        print(f"Model Instantiated. Total Parameters: {total_params:,}")
        print(f" - Frozen Parameters: {frozen_params:,}")
        print(f" - Trainable Parameters: {trainable_params:,}")

    def forward(self, x):
        # Extract features (B, 1280)
        x = self.backbone(x)
        # Pass through projection head
        x = self.projection_head(x)
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x

def export_onnx(model, path):
    """
    Exports the model to ONNX format.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
    torch.onnx.export(
        model, 
        dummy_input, 
        path, 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
    )
    print(f"Model exported to ONNX: {path}")

def export_torchscript(model, path):
    """
    Exports the model via torch.jit.trace.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(path)
    print(f"Model exported to TorchScript: {path}")
