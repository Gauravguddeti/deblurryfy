import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model.deblurgan_refined import load_deblur_model  # use classical refined pipeline model

def evaluate_deblurred_image(deblurred_path, sharp_path):
    """Evaluate the quality of deblurred image compared to sharp reference."""
    # Read images
    deblurred = cv2.imread(deblurred_path)
    sharp = cv2.imread(sharp_path)
    
    # Ensure the images have the same size
    if deblurred.shape != sharp.shape:
        deblurred = cv2.resize(deblurred, (sharp.shape[1], sharp.shape[0]))
    
    # Convert to grayscale for analysis
    deblurred_gray = cv2.cvtColor(deblurred, cv2.COLOR_BGR2GRAY)
    sharp_gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    ssim_score = ssim(deblurred_gray, sharp_gray)
    psnr_score = psnr(sharp_gray, deblurred_gray)
    
    # Calculate edge similarity
    edges_deblurred = cv2.Canny(deblurred_gray, 100, 200)
    edges_sharp = cv2.Canny(sharp_gray, 100, 200)
    edge_similarity = ssim(edges_deblurred, edges_sharp)
    
    # Calculate histogram similarity
    hist_sharp = cv2.calcHist([sharp], [0], None, [256], [0, 256])
    hist_deblurred = cv2.calcHist([deblurred], [0], None, [256], [0, 256])
    cv2.normalize(hist_sharp, hist_sharp, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_deblurred, hist_deblurred, 0, 1, cv2.NORM_MINMAX)
    hist_similarity = cv2.compareHist(hist_sharp, hist_deblurred, cv2.HISTCMP_CORREL)
    
    return {
        'ssim': ssim_score,
        'psnr': psnr_score,
        'edge_similarity': edge_similarity,
        'hist_similarity': hist_similarity
    }

def test_model():
    """Test DeblurGANv2 model on all test images."""
    print("\nTesting Refined Deblurring Pipeline")
    model = load_deblur_model()
     
    # Directory setup
    samples_dir = 'data/samples'
    output_dir = f'test_results/refined_pipeline'
    os.makedirs(output_dir, exist_ok=True)
     
    results = []
     
    # Find all blurred images and their sharp counterparts
    for file in os.listdir(samples_dir):
        if 'blurred' in file:
            base_name = file.replace('_blurred.jpg', '')
            sharp_file = f"{base_name}_sharp.jpg"
            
            if os.path.exists(os.path.join(samples_dir, sharp_file)):
                blurred_path = os.path.join(samples_dir, file)
                sharp_path = os.path.join(samples_dir, sharp_file)
                
                print(f"\nProcessing {file}...")
                
                # Load and process the image
                blurred_img = Image.open(blurred_path)
                deblurred_img = model.deblur_image(blurred_img)
                
                # Save the deblurred result
                output_path = os.path.join(output_dir, f"{base_name}_deblurred.jpg")
                deblurred_img.save(output_path, "JPEG", quality=95)
                print(f"Saved deblurred image to {output_path}")
                
                # Evaluate the result
                metrics = evaluate_deblurred_image(output_path, sharp_path)
                results.append({
                    'name': base_name,
                    'metrics': metrics
                })
                
                # Create comparison visualization
                compare_img_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
                create_comparison_image(blurred_path, output_path, sharp_path, compare_img_path)
    
    # Compute average metrics
    avg_metrics = {'ssim': 0.0, 'psnr': 0.0, 'edge_similarity': 0.0, 'hist_similarity': 0.0}
    for r in results:
        for k in avg_metrics:
            avg_metrics[k] += r['metrics'][k]
    count = len(results) if results else 1
    for k in avg_metrics:
        avg_metrics[k] /= count
    
    return results, avg_metrics

def create_comparison_image(blurred_path, deblurred_path, sharp_path, output_path):
    """Create a side-by-side comparison image."""
    # Read images
    blurred = cv2.imread(blurred_path)
    deblurred = cv2.imread(deblurred_path)
    sharp = cv2.imread(sharp_path)
    
    # Ensure all images have the same size (use sharp image size as reference)
    if blurred.shape != sharp.shape:
        blurred = cv2.resize(blurred, (sharp.shape[1], sharp.shape[0]))
    if deblurred.shape != sharp.shape:
        deblurred = cv2.resize(deblurred, (sharp.shape[1], sharp.shape[0]))
    
    # Convert to RGB for matplotlib
    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    deblurred_rgb = cv2.cvtColor(deblurred, cv2.COLOR_BGR2RGB)
    sharp_rgb = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
    
    # Create figure with three side-by-side images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(blurred_rgb)
    plt.title('Blurred')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(deblurred_rgb)
    plt.title('Deblurred')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sharp_rgb)
    plt.title('Reference Sharp')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Entry point: run the DeblurGANv2 tests and report results."""
    # Ensure base test_results directory exists
    os.makedirs('test_results', exist_ok=True)
    try:
        results, avg_metrics = test_model()
    except Exception as e:
        print(f"Error testing DeblurGANv2: {e}")
        return

    # Print per-image metrics
    print("\n\nDeblurGANv2 Evaluation:")
    print("-" * 60)
    print(f"{'Image':<10} {'SSIM':<10} {'PSNR':<10} {'Edge Sim':<10} {'Hist Sim':<10}")
    print("-" * 60)
    for r in results:
        m = r['metrics']
        print(f"{r['name']:<10} {m['ssim']:.4f}    {m['psnr']:.2f} dB    {m['edge_similarity']:.4f}    {m['hist_similarity']:.4f}")
    print("-" * 60)

    # Print average metrics
    print(f"{'Average':<10} {avg_metrics['ssim']:.4f}    {avg_metrics['psnr']:.2f} dB    {avg_metrics['edge_similarity']:.4f}    {avg_metrics['hist_similarity']:.4f}")
    
    # Save summary to file
    summary_path = os.path.join('test_results', 'refined_pipeline', 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("DeblurGANv2 Evaluation:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Image':<10} {'SSIM':<10} {'PSNR':<10} {'Edge Sim':<10} {'Hist Sim':<10}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            m = r['metrics']
            f.write(f"{r['name']:<10} {m['ssim']:.4f}    {m['psnr']:.2f} dB    {m['edge_similarity']:.4f}    {m['hist_similarity']:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Average':<10} {avg_metrics['ssim']:.4f}    {avg_metrics['psnr']:.2f} dB    {avg_metrics['edge_similarity']:.4f}    {avg_metrics['hist_similarity']:.4f}\n")
 
if __name__ == "__main__":
    main()
