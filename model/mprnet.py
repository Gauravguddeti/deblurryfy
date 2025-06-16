import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class MPRNetDeblurrer:
    """Wrapper for MPRNet deblurring network from Torch Hub."""
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pretrained MPRNet model from hub with explicit trust
        try:
            self.model = torch.hub.load(
                'swz30/MPRNet', 'MPRNet', pretrained=True, trust_repo=True
            ).to(self.device)
            self.model.eval()
            self.fallback = False
        except Exception as e:
            print(f"Error loading MPRNet from hub: {e}\nFalling back to RefinedDeblurringModel.")
            from model.deblurgan_refined import RefinedDeblurringModel
            self.model = RefinedDeblurringModel(device=self.device)
            self.fallback = True

        # Transform to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def deblur_image(self, image):
        # Ensure PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # If fallback to classical model, delegate
        if getattr(self, 'fallback', False):
            return self.model.deblur_image(image)

        # Preprocess for MPRNet
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        # Postprocess
        output = output.squeeze(0).cpu()
        output = torch.clamp(output, 0, 1)
        pil = transforms.ToPILImage()(output)
        return pil

@torch.no_grad()
def load_deblur_model():
    """Load MPRNet model for deblurring."""
    torch.set_grad_enabled(False)
    return MPRNetDeblurrer()
