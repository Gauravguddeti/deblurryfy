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
from urllib.parse import urlparse

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
torch.set_num_threads(1)


class LayerNorm(nn.Module):
    """Layer normalization for vision transformers"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention"""
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with MDTA and GDFN"""
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(channels, data_format='channels_first')
        self.attn = MDTA(channels, num_heads)
        self.norm2 = LayerNorm(channels, data_format='channels_first')
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding"""
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    """Downsampling module"""
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Upsampling module"""
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    """Restormer: Efficient Transformer for High-Resolution Image Restoration"""
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias'
    ):
        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(channels=dim, num_heads=heads[0], expansion_factor=ffn_expansion_factor) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(channels=int(dim*2**1), num_heads=heads[1], expansion_factor=ffn_expansion_factor) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(channels=int(dim*2**2), num_heads=heads[2], expansion_factor=ffn_expansion_factor) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(channels=int(dim*2**3), num_heads=heads[3], expansion_factor=ffn_expansion_factor) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(channels=int(dim*2**2), num_heads=heads[2], expansion_factor=ffn_expansion_factor) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(channels=int(dim*2**1), num_heads=heads[1], expansion_factor=ffn_expansion_factor) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(channels=int(dim*2**1), num_heads=heads[0], expansion_factor=ffn_expansion_factor) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(channels=int(dim*2**1), num_heads=heads[0], expansion_factor=ffn_expansion_factor) for i in range(num_refinement_blocks)])
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


class RestormerDeblurModel:
    """Restormer-based image deblurring model wrapper"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize Restormer model
        self.model = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False
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
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized weights.")
    
    def download_pretrained_weights(self, save_path="model/restormer_deblur.pth"):
        """Download pretrained Restormer weights for deblurring"""
        # Official Restormer deblurring weights
        model_url = "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            print(f"Downloading Restormer weights from {model_url}...")
            response = requests.get(model_url, stream=True)
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
            
            print(f"\n✓ Weights downloaded to {save_path}")
            return save_path
            
        except Exception as e:
            print(f"\n✗ Error downloading weights: {e}")
            print("The model will work with random weights but performance will be poor.")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image for Restormer input"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_size = image.size
        
        # Convert to tensor
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
        """Apply post-processing enhancements"""
        # Slight sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=110, threshold=2))
        
        # Subtle contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def deblur_image(self, input_image):
        """Deblur a single image using Restormer"""
        with torch.no_grad():
            # Preprocess
            input_tensor, original_size = self.preprocess_image(input_image)
            
            # Model inference
            output_tensor = self.model(input_tensor)
            
            # Postprocess
            deblurred_image = self.postprocess_image(output_tensor, original_size)
            
            # Apply enhancements
            deblurred_image = self.enhance_result(deblurred_image)
            
            return deblurred_image
    
    def save_deblurred_image(self, input_path, output_dir="outputs"):
        """Process and save deblurred image"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{input_name}_restormer_deblurred.jpg")
        
        # Process image
        deblurred_image = self.deblur_image(input_path)
        
        # Save with high quality
        deblurred_image.save(output_path, "JPEG", quality=98, optimize=True)
        print(f"Deblurred image saved to: {output_path}")
        
        return output_path


def load_deblur_model(model_path=None, download_weights=True):
    """Load Restormer deblurring model"""
    
    # Default path for weights
    if model_path is None:
        model_path = "model/restormer_deblur.pth"
    
    # Create model
    model = RestormerDeblurModel()
    
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
    print("Testing Restormer model...")
    model = load_deblur_model(download_weights=False)  # Set to True to download weights
    print("✓ Restormer model loaded successfully!")
    print(f"Device: {model.device}")
    
    # Test with sample if available
    sample_path = "data/samples/test1_blurred.jpg"
    if os.path.exists(sample_path):
        print(f"Testing with {sample_path}...")
        try:
            result = model.deblur_image(sample_path)
            output_path = "restormer_test_output.jpg"
            result.save(output_path, "JPEG", quality=95)
            print(f"✓ Test output saved to {output_path}")
        except Exception as e:
            print(f"✗ Test failed: {e}")
    else:
        print(f"Sample file {sample_path} not found. Model is ready for use.")
