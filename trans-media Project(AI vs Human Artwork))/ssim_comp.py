import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def load_image_as_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def apply_canny_edge_detector(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def calculate_ssim(image1, image2):
    return ssim(image1, image2)

def display_images(images, titles):
    plt.figure(figsize=(10, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

def main(image_path1, image_path2):
    img1 = load_image_as_grayscale(image_path1)
    img2 = load_image_as_grayscale(image_path2)

    # Resize images to the smallest dimensions among the two
    height, width = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1 = resize_image(img1, width, height)
    img2 = resize_image(img2, width, height)

    edges1 = apply_canny_edge_detector(img1, 100, 200)
    edges2 = apply_canny_edge_detector(img2, 100, 200)

    ssim_index = calculate_ssim(edges1, edges2)
    print(f"SSIM between the two edge-detected images: {ssim_index:.3f}")

    display_images([img1, img2, edges1, edges2], ['Original Image', 'AI-Generated Image', 'Original Edges', 'AI-Generated Edges'])

# Replace these paths with your actual image paths
main('/home/kunj/Downloads/cv/dgt/human/lightinside.jpeg', '/home/kunj/Downloads/cv/dgt/AI_generated/lightinside.jpeg')
