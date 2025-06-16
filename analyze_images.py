import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def analyze_image_pair(blurred_path, sharp_path):
    """Analyze differences between blurred and sharp images"""
    
    # Read images
    blurred = cv2.imread(blurred_path)
    sharp = cv2.imread(sharp_path)
    
    # Convert to RGB for display
    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    sharp_rgb = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
    
    # Calculate difference
    diff = cv2.absdiff(sharp, blurred)
    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for analysis
    blurred_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    sharp_gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    ssim_score = ssim(blurred_gray, sharp_gray)
    psnr_score = psnr(sharp_gray, blurred_gray)
    
    # Calculate edge differences
    edges_blurred = cv2.Canny(blurred_gray, 100, 200)
    edges_sharp = cv2.Canny(sharp_gray, 100, 200)
    edges_diff = cv2.absdiff(edges_sharp, edges_blurred)
    
    # FFT analysis to see differences in frequency domain
    fft_blurred = np.fft.fftshift(np.fft.fft2(blurred_gray))
    fft_sharp = np.fft.fftshift(np.fft.fft2(sharp_gray))
    
    magnitude_blurred = 20 * np.log(np.abs(fft_blurred) + 1)
    magnitude_sharp = 20 * np.log(np.abs(fft_sharp) + 1)
    
    # Normalize for display
    magnitude_blurred = magnitude_blurred / np.max(magnitude_blurred) * 255
    magnitude_sharp = magnitude_sharp / np.max(magnitude_sharp) * 255
    
    # Print analysis results
    print(f"Analysis for {os.path.basename(blurred_path)} and {os.path.basename(sharp_path)}")
    print(f"SSIM Score: {ssim_score:.4f} (higher is better, 1.0 is perfect match)")
    print(f"PSNR: {psnr_score:.2f} dB (higher is better)")
    
    # Display images
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(blurred_rgb)
    plt.title('Blurred Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(sharp_rgb)
    plt.title('Sharp Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(diff_rgb)
    plt.title('Difference')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(edges_blurred, cmap='gray')
    plt.title('Edges (Blurred)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(edges_sharp, cmap='gray')
    plt.title('Edges (Sharp)')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(edges_diff, cmap='gray')
    plt.title('Edge Difference')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"analysis_{os.path.basename(blurred_path).split('.')[0]}.png")
    plt.close()
    
    return {
        'ssim': ssim_score,
        'psnr': psnr_score,
        'blurred': blurred,
        'sharp': sharp,
        'diff': diff
    }

def main():
    samples_dir = 'data/samples'
    results = []
    
    # Find all blurred images and their sharp counterparts
    for file in os.listdir(samples_dir):
        if 'blurred' in file:
            base_name = file.replace('_blurred.jpg', '')
            sharp_file = f"{base_name}_sharp.jpg"
            
            if os.path.exists(os.path.join(samples_dir, sharp_file)):
                blurred_path = os.path.join(samples_dir, file)
                sharp_path = os.path.join(samples_dir, sharp_file)
                
                result = analyze_image_pair(blurred_path, sharp_path)
                results.append({
                    'name': base_name,
                    'metrics': result
                })
    
    # Print summary
    print("\nSummary of Image Analysis:")
    print("--------------------------")
    for result in results:
        print(f"{result['name']}: SSIM={result['metrics']['ssim']:.4f}, PSNR={result['metrics']['psnr']:.2f}dB")

if __name__ == "__main__":
    main()
