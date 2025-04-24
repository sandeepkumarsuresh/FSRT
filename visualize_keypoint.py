"""
Author: Sandeep Kumar Suresh

    EE23S059

The below code is used to extract keypoints from the face alignment 
module and the keypoint detection module used in the paper to plot the 
keypoints on the image.

"""

## The below code is to visualize face alignment keypoints 


# import face_alignment
# from skimage import io
# import matplotlib.pyplot as plt
# import numpy as np  
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

# input = io.imread('/4TBHD/fsrt/source_img/sandy_specs.jpg')
# preds = fa.get_landmarks(input)
# print(np.array(preds).shape)  


# # Plot image and keypoints
# if preds is not None:
#     plt.imshow(input)
#     for i, pred in enumerate(preds):  # In case multiple faces are detected
#         plt.scatter(pred[:, 0], pred[:, 1], s=10, label=f'Face {i+1}')
#     plt.legend()
#     plt.title("Facial Keypoints")
#     plt.axis('off')
#     plt.show()
#     plt.savefig('keypoints_specs.png', bbox_inches='tight', dpi=300)
#     plt.close()
# else:
#     print("No face detected.")

    

import matplotlib.pyplot as plt
import numpy as np
import torch
from modules.keypoint_detector import KPDetector
import torchvision.transforms as transforms
from PIL import Image


def plot_keypoints(image_tensor, keypoints, title="Keypoints", save_path='keypoint.png'):



    # Convert image tensor to numpy and transpose to (H, W, 3)
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    print("Image shape:", image.shape)  
    image = (image - image.min()) / (image.max() - image.min())  # Normalize for visualization


    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)

    # Denormalize keypoints from [-1, 1] to image pixel space
    h, w = image.shape[:2]
    kp = keypoints.detach().cpu().numpy()
    print(kp)
    kp = (kp + 1) / 2  # from [-1, 1] to [0, 1]
    kp[:, 0] *= w
    kp[:, 1] *= h

    # Plot keypoints
    plt.scatter(kp[:, 0], kp[:, 1], c='r', s=40, marker='x')

    if save_path:
        plt.savefig(save_path)
    plt.axis('off')
    plt.show()


def load_image(path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: (1, 3, H, W)


# def visualize_keypoints(image, keypoints, heatmaps, scale_factor=1.0):
#     """
#     Visualize detected keypoints and their heatmaps.
    
#     Args:
#         image (torch.Tensor): Original image tensor (B, C, H, W)
#         keypoints (torch.Tensor): Detected keypoints (B, num_kp, 2)
#         heatmaps (torch.Tensor): Heatmaps from KPDetector (B, num_kp, 1, H, W)
#         scale_factor (float): Scaling factor used in KPDetector
#     """
#     # Convert tensors to numpy arrays
#     img = image[0].permute(1, 2, 0).cpu().detach().numpy()
#     img = np.clip(img, 0, 1)  # Ensure valid RGB range
#     kps = keypoints[0].cpu().detach().numpy()
#     heatmaps_np = heatmaps[0].squeeze(1).cpu().detach().numpy()


#     print("Original image shape:", image.shape)  # Tensor shape
#     print("Image after permute:", img.shape)     # NumPy shape
#     print("Heatmap shape:", heatmaps_np.shape)   # (num_kp, H, W)



#     print("Keypoints:", kps)
#     plot_heatmaps_per_kp(heatmaps_np)


#     # Get dimensions
#     img_h, img_w = img.shape[:2]
#     heatmap_h, heatmap_w = heatmaps_np.shape[1:]

#     # # Convert keypoints to image coordinates (critical fix)
#     # kps[:, 0] = kps[:, 0] * (img_w / heatmap_w)
#     # kps[:, 1] = kps[:, 1] * (img_h / heatmap_h)

#     # kps[:, 0] = ((kps[:, 0] + 1) / 2) * img_w
#     # kps[:, 1] = ((kps[:, 1] + 1) / 2) * img_h

#     # # Step 1: From [-1, 1] to heatmap space
#     # kps[:, 0] = ((kps[:, 0] + 1) / 2) * heatmap_w
#     # kps[:, 1] = ((kps[:, 1] + 1) / 2) * heatmap_h

#     # # Step 2: Upscale from heatmap to image space
#     # kps[:, 0] *= (img_w / heatmap_w)
#     # kps[:, 1] *= (img_h / heatmap_h)

#     # If keypoints are in heatmap coordinates (e.g., 58x58 space), convert:
#     kps[:, 0] = kps[:, 0] * (img_w / heatmap_w)
#     kps[:, 1] = kps[:, 1] * (img_h / heatmap_h)



#     print("Keypoints:", kps)

#     # Create figure with proper layout
#     fig = plt.figure(figsize=(16, 8))
    
#     # Image with keypoints subplot
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax1.imshow(img)
#     ax1.scatter(kps[:, 0], kps[:, 1], c='cyan', s=80, 
#                 edgecolors='black', linewidths=2, alpha=0.8)
#     ax1.set_title('Keypoints on Image')
#     ax1.axis('off')

#     # Heatmaps subplot
#     ax2 = fig.add_subplot(1, 2, 2)
#     heatmap_grid = heatmaps_np.transpose(1, 2, 0).reshape(-1, heatmap_w)
#     heatmap_plot = ax2.imshow(heatmap_grid, cmap='hot', aspect='auto')
#     plt.colorbar(heatmap_plot, ax=ax2, shrink=0.8)
#     ax2.set_title('Heatmap Activation Patterns')
#     ax2.axis('off')

#     plt.tight_layout()
#     plt.show()

# def plot_heatmaps_per_kp(heatmaps_np):
#     num_kp = heatmaps_np.shape[0]
#     cols = 5
#     rows = (num_kp + cols - 1) // cols

#     fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
#     for i in range(rows * cols):
#         ax = axes[i // cols, i % cols]
#         if i < num_kp:
#             ax.imshow(heatmaps_np[i], cmap='hot')
#             ax.set_title(f"KP {i}")
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('heatmaps_per_kp.png', bbox_inches='tight', dpi=300)
#     plt.close()


if __name__ == "__main__":

    image_path = '/4TBHD/fsrt/source_img/squint_eye.jpeg'
    image_tensor = load_image(image_path)


    # Load model
    model = KPDetector()
    model.eval()

    # Forward pass
    with torch.no_grad():
        keypoints, _ = model(image_tensor)
    
    print(type(_))
    print(_['feature_map'].shape)  # Feature map shape
    print(_['heatmap'].shape)  # Heatmap shape

    # Visualize
    # plot_keypoints(image_tensor[0], keypoints[0], title="Predicted Keypoints")

    # visualize_keypoints(image_tensor, keypoints, _['heatmap'], scale_factor=0.25)