import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.deblurgan import DeblurGANv2Generator


class BlurDataset(Dataset):
    """Dataset for blur/sharp image pairs."""
    
    def __init__(self, blur_dir, sharp_dir, transform=None, img_size=256):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.img_size = img_size
        
        # Get list of image files
        self.blur_images = sorted([f for f in os.listdir(blur_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.sharp_images = sorted([f for f in os.listdir(sharp_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure we have matching pairs
        assert len(self.blur_images) == len(self.sharp_images), \
            "Number of blur and sharp images must match"
    
    def __len__(self):
        return len(self.blur_images)
    
    def __getitem__(self, idx):
        # Load images
        blur_path = os.path.join(self.blur_dir, self.blur_images[idx])
        sharp_path = os.path.join(self.sharp_dir, self.sharp_images[idx])
        
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        
        # Resize images
        blur_img = blur_img.resize((self.img_size, self.img_size), Image.LANCZOS)
        sharp_img = sharp_img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Apply transforms
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        
        return blur_img, sharp_img


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use VGG19 features
        from torchvision.models import vgg19
        vgg = vgg19(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16])
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return nn.functional.mse_loss(x_vgg, y_vgg)


class DeblurTrainer:
    """Training class for DeblurGAN-v2."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.generator = DeblurGANv2Generator().to(self.device)
        
        # Initialize losses
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        
        # Initialize optimizer
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=config.get('lr', 0.0001),
            betas=(0.5, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G, 
            step_size=config.get('lr_decay_steps', 50),
            gamma=config.get('lr_decay_gamma', 0.5)
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.generator.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for blur_imgs, sharp_imgs in pbar:
            blur_imgs = blur_imgs.to(self.device)
            sharp_imgs = sharp_imgs.to(self.device)
            
            # Generator forward pass
            self.optimizer_G.zero_grad()
            generated_imgs = self.generator(blur_imgs)
            
            # Calculate losses
            mse_loss = self.mse_loss(generated_imgs, sharp_imgs)
            perceptual_loss = self.perceptual_loss(generated_imgs, sharp_imgs)
            
            # Total loss
            total_g_loss = mse_loss + 0.1 * perceptual_loss
            
            # Backward pass
            total_g_loss.backward()
            self.optimizer_G.step()
            
            total_loss += total_g_loss.item()
            pbar.set_postfix({'Loss': f'{total_g_loss.item():.4f}'})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model."""
        self.generator.eval()
        total_loss = 0
        
        with torch.no_grad():
            for blur_imgs, sharp_imgs in dataloader:
                blur_imgs = blur_imgs.to(self.device)
                sharp_imgs = sharp_imgs.to(self.device)
                
                generated_imgs = self.generator(blur_imgs)
                mse_loss = self.mse_loss(generated_imgs, sharp_imgs)
                total_loss += mse_loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch, loss, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'scheduler_G': self.scheduler_G.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        return checkpoint['epoch'], checkpoint['loss']


def train_deblur_model(config):
    """Main training function."""
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    train_dataset = BlurDataset(
        blur_dir=config['train_blur_dir'],
        sharp_dir=config['train_sharp_dir'],
        transform=transform,
        img_size=config.get('img_size', 256)
    )
    
    val_dataset = BlurDataset(
        blur_dir=config['val_blur_dir'],
        sharp_dir=config['val_sharp_dir'],
        transform=transform,
        img_size=config.get('img_size', 256)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # Initialize trainer
    trainer = DeblurTrainer(config)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.get('num_epochs', 100)):
        print(f"\nEpoch {epoch+1}/{config.get('num_epochs', 100)}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        # Update learning rate
        trainer.scheduler_G.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                epoch, val_loss, 
                os.path.join(config['save_dir'], 'best_model.pth')
            )
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                epoch, val_loss,
                os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            )


if __name__ == "__main__":
    # Example training configuration
    config = {
        'train_blur_dir': 'data/train/blur',
        'train_sharp_dir': 'data/train/sharp',
        'val_blur_dir': 'data/val/blur',
        'val_sharp_dir': 'data/val/sharp',
        'save_dir': 'model',
        'num_epochs': 100,
        'batch_size': 8,
        'lr': 0.0001,
        'lr_decay_steps': 50,
        'lr_decay_gamma': 0.5,
        'img_size': 256,
        'num_workers': 4
    }
    
    print("DeblurGAN-v2 Training Script")
    print("This script is ready for training when you provide dataset directories.")
    print("Update the config above with your dataset paths and run this script.")
    
    # Uncomment the line below to start training
    # train_deblur_model(config)
