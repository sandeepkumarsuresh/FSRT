"""
Author: Sandeep Kumar Suresh
        EE23S059


Code for ploting keypoints and heatmaps

"""


import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from modules.keypoint_detector import KPDetector


def plot_keypoints_on_image(image, keypoints):
    img = image.squeeze().permute(1, 2, 0).cpu().numpy()
    kp = keypoints.squeeze().cpu().numpy()  # shape: [num_kp, 2]

    plt.imshow(img)
    plt.scatter(kp[:, 0] * img.shape[1], kp[:, 1] * img.shape[0], c='r', s=40)
    plt.title("Image with Keypoints")
    plt.axis('off')
    plt.savefig('keypoints_on_image.png')
    plt.show()


def plot_heatmaps(heatmaps):
    heatmaps = heatmaps.squeeze(0).squeeze(1).cpu()  # shape: [num_kp, H, W]

    num_kp = heatmaps.shape[0]
    cols = 5
    rows = (num_kp + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))
    for i in range(num_kp):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(heatmaps[i], cmap='hot')
        plt.title(f"Keypoint {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('heatmaps.png')
    plt.show()

def load_image(path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: (1, 3, H, W)


if __name__ == '__main__':

    image_path = '/4TBHD/fsrt/source_img/7276470@N03_identity_3@3792415751_0.jpg'
    image = load_image(image_path)

    model = KPDetector(num_kp=10)
    model.eval()

    with torch.no_grad():
        keypoints, out_dict = model(image)
        heatmaps = out_dict['heatmap'] 

    plot_keypoints_on_image(image, keypoints)
    plot_heatmaps(heatmaps)
