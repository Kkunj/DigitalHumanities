import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def resize_images(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)
    return cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA), cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

def apply_canny_edge_detector(image, low_threshold=50, high_threshold=150):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return edges

def calculate_ssim(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def find_dominant_colors(image, num_colors=5):
    data = np.reshape(image, (-1, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(data)
    colors = kmeans.cluster_centers_
    return colors

def plot_color_histogram(image, ax):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(histr, color=col)
        ax.set_xlim([0, 256])

def plot_dominant_colors(colors, ax):
    bar = np.zeros((50, 300, 3), dtype='uint8')
    step = 300 // colors.shape[0]
    for i in range(colors.shape[0]):
        bar[:, i*step:(i+1)*step, :] = colors[i]
    ax.imshow(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
    ax.axis('off')

def main(image_path1, image_path2):
    img1 = load_image(image_path1)
    img2 = load_image(image_path2)

    img1, img2 = resize_images(img1, img2)  # Resize images to the same dimensions

    edges1 = apply_canny_edge_detector(img1)
    edges2 = apply_canny_edge_detector(img2)
    ssim_index = calculate_ssim(img1, img2)
    colors1 = find_dominant_colors(img1)
    colors2 = find_dominant_colors(img2)

    fig, axs = plt.subplots(3, 4, figsize=(20, 12))

    axs[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 1].set_title('AI-Generated Image')
    for ax in axs[0, :2]:
        ax.axis('off')

    plot_dominant_colors(colors1, axs[0, 2])
    plot_dominant_colors(colors2, axs[0, 3])
    axs[0, 2].set_title('Dominant Colors - Original')
    axs[0, 3].set_title('Dominant Colors - AI-Generated')

    plot_color_histogram(img1, axs[1, 0])
    plot_color_histogram(img2, axs[1, 1])
    axs[1, 0].set_title('Color Histogram - Original')
    axs[1, 1].set_title('Color Histogram - AI-Generated')

    axs[1, 2].imshow(edges1, cmap='gray')
    axs[1, 3].imshow(edges2, cmap='gray')
    axs[1, 2].set_title('Edges - Original')
    axs[1, 3].set_title('Edges - AI-Generated')
    for ax in axs[1, 2:]:
        ax.axis('off')

    axs[2, 0].text(0.5, 0.5, f'SSIM Score: {ssim_index:.3f}', fontsize=16, ha='center', va='center')
    axs[2, 0].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main('/home/kunj/Downloads/cv/dgt/human/lightinside.jpeg', '/home/kunj/Downloads/cv/dgt/AI_generated/lightinside.jpeg')
