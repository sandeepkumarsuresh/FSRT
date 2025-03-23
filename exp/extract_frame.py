import cv2
import numpy as np
import matplotlib.pyplot as plt
def extract_first_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"First frame saved at: {output_path}")
    else:
        print("Error: Could not read the first frame.")
    
    cap.release()


def extract_and_stack_frames(video_path, output_path, num_frames=5, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    frames = []
    count = 0
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
            frames.append(frame)
        
        count += 1
    
    cap.release()
    
    if len(frames) < num_frames:
        print("Warning: Not enough frames were extracted.")
    
    stacked_image = np.vstack(frames)  # Stack frames horizontally
    
    plt.figure(figsize=(num_frames * 2, 4))
    # plt.axis('off')
    plt.imshow(stacked_image)
    plt.xticks([stacked_image.shape[0] // 8,  stacked_image.shape[0] // 8], ["Driving Video", "Driving video  Source frame"])
    # plt.yticks([])  # Remove y-axis labels

    # Set y-axis labels to frame numbers
    frame_indices = np.linspace(0, stacked_image.shape[0] - 1, num_frames, dtype=int)
    plt.yticks(frame_indices, labels=[str(i) for i in range(num_frames)])


    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Stacked frames saved at: {output_path}")



# Example usage
extract_and_stack_frames("/4TBHD/fsrt/absolute_motion_transfer1.mp4", "absolute_motion_transfer1.png", num_frames=10, frame_interval=20)



# extract_first_frame("/4TBHD/fsrt/driving_video/00010.mp4", "source_00010.jpg")
