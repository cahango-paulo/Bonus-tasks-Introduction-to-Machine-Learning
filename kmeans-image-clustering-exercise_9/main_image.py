import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def kmeans_segmentation(image_array, n_clusters, max_iter=100):

    pixels = image_array.reshape(-1, image_array.shape[-1]).astype(np.float32)
    
    np.random.seed(42)
    centroids = pixels[np.random.choice(pixels.shape[0], n_clusters, replace=False)]
    
    for _ in range(max_iter):
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([pixels[labels == k].mean(axis=0) for k in range(n_clusters)])
        
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
    
    segmented_pixels = centroids[labels].astype(np.uint8)
    segmented_image = segmented_pixels.reshape(image_array.shape)
    
    return segmented_image, labels.reshape(image_array.shape[:2])

def apply_mask(image_array, mask):
    masked_image = np.zeros_like(image_array)
    for channel in range(image_array.shape[2]):
        masked_image[..., channel] = image_array[..., channel] * mask
    return masked_image

def main():
    # Load the input image
    image_path = "img1.png"  
    original_image = Image.open(image_path)
    image_array = np.array(original_image)
    
    n_clusters = 4  
    
    segmented_image, cluster_mask = kmeans_segmentation(image_array, n_clusters)
    
    cluster_masks = []
    for i in range(n_clusters):
        mask = (cluster_mask == i).astype(np.uint8)
        cluster_masks.append(mask)
    
    masked_images = [apply_mask(image_array, mask) for mask in cluster_masks]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f"Segmented Image (K={n_clusters})")
    plt.axis('off')
    
    for i in range(n_clusters):
        plt.subplot(2, n_clusters, n_clusters + i + 1)
        plt.imshow(masked_images[i])
        plt.title(f"Segment {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
