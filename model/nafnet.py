import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import os
import warnings
import requests

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
torch.set_num_threads(1)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetDeblurModel:
    """NAFNet-based image deblurring model wrapper"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize NAFNet model - optimized for deblurring
        self.model = NAFNet(
            img_channel=3,
            width=64,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2]
        ).to(self.device)
        
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.ToTensor()
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pretrained weights provided. Using randomly initialized model.")
    
    def load_model(self, model_path):
        """Load pretrained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
              self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized weights.")
    
    def download_pretrained_weights(self, save_path="model/nafnet_deblur.pth"):
        """Download pretrained NAFNet weights for deblurring"""
        # Try multiple sources for NAFNet weights
        model_urls = [
            "https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-GoPro-width64.pth",
            "https://drive.google.com/uc?id=14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X",  # Alternative source
            "https://huggingface.co/spaces/akhaliq/NAFNet/resolve/main/experiments/pretrained_models/NAFNet-GoPro-width64.pth"
        ]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for i, model_url in enumerate(model_urls):
            try:
                print(f"Attempting download from source {i+1}...")
                print(f"URL: {model_url}")
                
                response = requests.get(model_url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownloading... {percent:.1f}%", end='', flush=True)
                
                print(f"\n✓ NAFNet weights downloaded to {save_path}")
                return save_path
                
            except Exception as e:
                print(f"\n✗ Source {i+1} failed: {e}")
                if i < len(model_urls) - 1:
                    print("Trying next source...")
                continue
        
        print("\n✗ All download sources failed.")
        print("You can manually download weights from:")
        print("https://github.com/megvii-research/NAFNet")
        print("Or use the model without pretrained weights (reduced performance)")
        return None
    
    def preprocess_image(self, image):
        """Preprocess image for NAFNet input"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_size = image.size
        
        # Convert to tensor [0, 1]
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return tensor, original_size
    
    def postprocess_image(self, tensor, original_size):
        """Convert model output back to PIL image"""
        # Clamp values to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        image = transforms.ToPILImage()(tensor.squeeze(0).cpu())
        
        # Resize to original size if needed
        if image.size != original_size:
            image = image.resize(original_size, Image.LANCZOS)
        
        return image
    
    def enhance_result(self, image):
        """Apply minimal post-processing to preserve color accuracy"""
        # Very light sharpening only
        image = image.filter(ImageFilter.UnsharpMask(radius=0.3, percent=105, threshold=1))
        return image
    
    def deblur_image(self, input_image):
        """Deblur a single image using NAFNet"""
        with torch.no_grad():
            # Preprocess
            input_tensor, original_size = self.preprocess_image(input_image)
            
            # Model inference
            output_tensor = self.model(input_tensor)
            
            # Postprocess
            deblurred_image = self.postprocess_image(output_tensor, original_size)
            
            # Light enhancement
            deblurred_image = self.enhance_result(deblurred_image)
            
            return deblurred_image
    
    def save_deblurred_image(self, input_path, output_dir="outputs"):
        """Process and save deblurred image"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{input_name}_nafnet_deblurred.jpg")
        
        # Process image
        deblurred_image = self.deblur_image(input_path)
        
        # Save with high quality
        deblurred_image.save(output_path, "JPEG", quality=98, optimize=True)
        print(f"Deblurred image saved to: {output_path}")
        
        return output_path


def load_deblur_model(model_path=None, download_weights=True):
    """Load NAFNet deblurring model"""
    
    # Default path for weights
    if model_path is None:
        model_path = "model/nafnet_deblur.pth"
    
    # Create model
    model = NAFNetDeblurModel()
    
    # Download weights if they don't exist and download is requested
    if download_weights and not os.path.exists(model_path):
        print("Pretrained weights not found. Downloading...")
        downloaded_path = model.download_pretrained_weights(model_path)
        if downloaded_path:
            model.load_model(downloaded_path)
    elif os.path.exists(model_path):
        model.load_model(model_path)
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing NAFNet model...")
    model = load_deblur_model(download_weights=False)  # Set to True to download weights
    print("✓ NAFNet model loaded successfully!")
    print(f"Device: {model.device}")
    
    # Test with sample if available
    sample_path = "data/samples/test1_blurred.jpg"
    if os.path.exists(sample_path):
        print(f"Testing with {sample_path}...")
        try:
            result = model.deblur_image(sample_path)
            output_path = "nafnet_test_output.jpg"
            result.save(output_path, "JPEG", quality=95)
            print(f"✓ Test output saved to {output_path}")
        except Exception as e:
            print(f"✗ Test failed: {e}")
    else:
        print(f"Sample file {sample_path} not found. Model is ready for use.")
