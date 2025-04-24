import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_first_frame(video_path, output_path, frame_index=0):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_index} saved at: {output_path}")
    else:
        print(f"Error: Could not read frame {frame_index}.")
    
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
    plt.xticks([stacked_image.shape[0] // 7,  stacked_image.shape[0] // 7], ["Driving Video", "Driving video  Source frame"])
    # plt.yticks([])  # Remove y-axis labels

    # Set y-axis labels to frame numbers
    frame_indices = np.linspace(0, stacked_image.shape[0] - 1, num_frames, dtype=int)
    plt.yticks(frame_indices, labels=[str(i) for i in range(num_frames)])


    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Stacked frames saved at: {output_path}")


if __name__ == '__main__':

    extract_and_stack_frames("/4TBHD/fsrt/results/exp_sunglass_2.mp4", "sunglass", num_frames=10, frame_interval=20)



    # extract_first_frame("/4TBHD/fsrt/driving_video/test1.mp4", "sandy_specs_21st_frame.jpg",frame_index=21)
