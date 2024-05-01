import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_image(image_path):
    return Image.open(image_path)

def plot_histogram(image, title):
    # Convert image to numpy array
    data = np.array(image)
    # Flatten the data to 1D array for each color channel
    r, g, b = data[:,:,0].flatten(), data[:,:,1].flatten(), data[:,:,2].flatten()
    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(r, bins=256, color='red', alpha=0.5, label='Red Channel')
    ax.hist(g, bins=256, color='green', alpha=0.5, label='Green Channel')
    ax.hist(b, bins=256, color='blue', alpha=0.5, label='Blue Channel')
    ax.legend()
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Color Histogram - {title}')
    plt.show()

def find_dominant_colors(image, num_colors):
    # Resize image to reduce the number of pixels for KMeans
    resized_image = image.resize((100, 100))
    data = np.array(resized_image)
    # Reshape the data to 2D (list of pixels)
    data = data.reshape((-1, 3))
    # Fit KMeans to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(data)
    colors = kmeans.cluster_centers_
    # Convert colors to integer from float
    colors = colors.round(0).astype(int)
    return colors

def plot_dominant_colors(colors, title):
    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(5, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    ax.imshow([colors], aspect='auto')
    ax.set_title(f'Dominant Colors - {title}')
    plt.show()

# Load images
image1 = load_image('/home/kunj/Documents/Twine/Stories/15.jpg')
image2 = load_image('/home/kunj/Documents/Twine/Stories/15_ai.jpeg')

# Plot histograms
plot_histogram(image1, 'Image 1')
plot_histogram(image2, 'Image 2')

# Get dominant colors
dominant_colors_image1 = find_dominant_colors(image1, 5)
dominant_colors_image2 = find_dominant_colors(image2, 5)

# Plot dominant colors
plot_dominant_colors(dominant_colors_image1, 'Image 1')
plot_dominant_colors(dominant_colors_image2, 'Image 2')
