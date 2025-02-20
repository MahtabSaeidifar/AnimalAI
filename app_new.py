# app_new.py
# Streamlit version = 1.19.0
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from streamlit_drawable_canvas import st_canvas
from sam2.build_sam import build_sam2_video_predictor
from hydra.core.global_hydra import GlobalHydra

# Ensure that logging messages from build_sam.py are displayed in Streamlit
import logging
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
def extract_frames(video_path, time_interval):
    """
    Extract frames from a video at a specified time interval.
    Returns (saved_frames, frame_dir, frame_rate).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None, None, None

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate <= 0:
        st.error("Error: Could not retrieve frame rate.")
        cap.release()
        return None, None, None

    frame_step = max(int(frame_rate * time_interval), 1)
    frame_count = 0
    saved_frames = []

    frame_dir = 'extracted_frames'
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_step == 0:
            frame_path = os.path.join(frame_dir, f"{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
        frame_count += 1

    cap.release()
    return saved_frames, frame_dir, frame_rate

def recommend_time_interval(video_duration):
    """Provide recommendations based on video length."""
    if video_duration < 60:
        st.write("For videos less than 1 minute, we recommend an interval between 0.1-0.5 seconds.")
    elif video_duration < 300:
        st.write("For videos between 1 and 5 minutes, we recommend 0.5-1 seconds.")
    elif video_duration < 600:
        st.write("For videos between 5 and 10 minutes, we recommend 1-1.5 seconds.")
    elif video_duration < 900:
        st.write("For videos between 10 and 15 minutes, we recommend 1.5-2 seconds.")
    elif video_duration < 1200:
        st.write("For videos between 15 and 20 minutes, we recommend 2-2.5 seconds.")
    elif video_duration < 1500:
        st.write("For videos between 20 and 25 minutes, we recommend 2.5-3 seconds.")
    elif video_duration < 1800:
        st.write("For videos between 25 and 30 minutes, we recommend 3-3.5 seconds.")
    elif video_duration < 2100:
        st.write("For videos between 30 and 35 minutes, we recommend 3.5-4 seconds.")
    elif video_duration < 2400:
        st.write("For videos between 35 and 40 minutes, we recommend 4-4.5 seconds.")
    elif video_duration < 2700:
        st.write("For videos between 40 and 45 minutes, we recommend 4.5-5 seconds.")
    elif video_duration < 3000:
        st.write("For videos between 45 and 50 minutes, we recommend 5-5.5 seconds.")
    elif video_duration < 3300:
        st.write("For videos between 50 and 55 minutes, we recommend 5.5-6 seconds.")
    elif video_duration < 3600:
        st.write("For videos between 55 and 60 minutes, we recommend 6-6.5 seconds.")
    else:
        st.write("For videos longer than 1 hour, please trim your video to smaller parts for efficiency.")

def show_mask(mask, frame):
    """Overlay mask onto the frame."""
    mask = mask.squeeze()
    mask_image = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask_image[..., 3] = (mask * 255).astype(np.uint8)  # Alpha channel
    mask_pil = Image.fromarray(mask_image, 'RGBA')
    result = Image.alpha_composite(frame.convert('RGBA'), mask_pil)
    return result

def calculate_activity_index(mask_dir):
    """
    Calculate the normalized activity index over time based on mask differences.
    Returns a list of activity_index values.
    """
    frame_names = sorted(os.listdir(mask_dir))
    activity_index = []
    
    prev_frame = None

    for i, frame_name in enumerate(frame_names):
        frame_path = os.path.join(mask_dir, frame_name)
        current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        if current_frame is None:
            st.warning(f"Warning: Could not read frame {frame_path}. Skipping.")
            continue

        if prev_frame is not None:
            if np.sum(current_frame > 0) == 0 or np.sum(prev_frame > 0) == 0:
                prev_frame = current_frame
                continue

            frame_diff = cv2.absdiff(current_frame, prev_frame)
            diff_pixel_count = np.sum(frame_diff > 0)

            pixel_count_current = np.sum(current_frame > 0)
            pixel_count_previous = np.sum(prev_frame > 0)
            total_pixel_count = pixel_count_current + pixel_count_previous

            normalized_activity_value = diff_pixel_count / total_pixel_count if total_pixel_count > 0 else 0
            activity_index.append(normalized_activity_value)
        
        prev_frame = current_frame

    return activity_index

def plot_activity_index(activity_index, frame_interval_seconds, video_duration):
    """Plot the normalized activity index over time (y-axis fixed between 0 and 1)."""
    num_activity_points = len(activity_index)
    time_seconds = np.arange(0, num_activity_points * frame_interval_seconds, frame_interval_seconds)
    if len(time_seconds) > len(activity_index):
        time_seconds = time_seconds[:len(activity_index)]

    # Decide tick intervals based on duration
    if video_duration <= 60:
        x_tick_interval = 2
    elif video_duration <= 300:
        x_tick_interval = 10
    elif video_duration <= 600:
        x_tick_interval = 20
    elif video_duration <= 3600:
        x_tick_interval = 120
    else:
        x_tick_interval = 100

    x_ticks = np.arange(0, video_duration + x_tick_interval, x_tick_interval)

    def format_time(x, _):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f'{minutes}:{seconds:02d}'

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_seconds, activity_index, marker='o', linestyle='-')
    ax.set_title('Normalized Activity Index Over Time')
    ax.set_xlabel('Time (minutes:seconds)')
    ax.set_ylabel('Normalized Activity Index')
    
    # Fix y-axis range to [0, 1] and set ticks at 0.1 intervals
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    return fig


# ----------------------------------------------------
# Streamlit Application
# ----------------------------------------------------
st.title("AnimalAI: An Open-Source Web Platform for Automated Animal Activity Index Calculation Using Interactive Deep Learning Segmentation")

# Initialize session state variables
if 'saved_frames' not in st.session_state:
    st.session_state.saved_frames = []
if 'frame_dir' not in st.session_state:
    st.session_state.frame_dir = ''
if 'frame_rate' not in st.session_state:
    st.session_state.frame_rate = 0

# Cached function to initialize the predictor
@st.cache_resource
def get_predictor(config_path, config_name, sam2_checkpoint, device):
    """
    Initialize and return the SAM2 video predictor model.
    Cached to prevent re-initialization on every Streamlit rerun.
    """
    try:
        predictor = build_sam2_video_predictor(
            config_path=config_path,
            config_name=config_name,
            ckpt_path=sam2_checkpoint,
            device=device,
            mode="eval",
            hydra_overrides_extra=[],  # Add any additional overrides if needed
            apply_postprocessing=True
        )
        return predictor
    except Exception as e:
        st.error(f"Error initializing the predictor: {e}")
        raise

def extract_points_from_canvas(canvas_data, scale_factor_w, scale_factor_h):
    """
    Extract circle center points from the st_canvas JSON data.
    Convert from scaled coordinates back to original (full-size) coordinates 
    using the provided scale factors.
    """
    object_points = []
    if canvas_data is not None and "objects" in canvas_data:
        for obj in canvas_data["objects"]:
            # We're only interested in circles (the user draws them)
            if obj["type"] == "circle":
                radius = obj["width"] / 2
                cx_scaled = obj["left"] + radius
                cy_scaled = obj["top"] + radius

                # Convert to the original (unscaled) coordinates
                cx_original = cx_scaled * scale_factor_w
                cy_original = cy_scaled * scale_factor_h

                # Round to int if needed
                object_points.append((int(cx_original), int(cy_original), 1))
            else:
                # Handle other shapes or ignore
                pass
    return object_points

def main():
    st.write("Current Working Directory:", os.getcwd())

    # Video uploader
    uploaded_video = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        # Save the uploaded file to a temporary path
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Analyze the video length
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Error: Could not open the uploaded video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / frame_rate if frame_rate > 0 else 0
        cap.release()

        st.write(f"Video Duration: {video_duration:.2f} seconds")
        recommend_time_interval(video_duration)

        frame_interval_seconds = st.number_input(
            "Enter the frame interval in seconds:",
            min_value=0.1,
            value=0.5,
            step=0.1
        )

        if st.button("Extract Frames"):
            with st.spinner("Extracting frames..."):
                saved_frames, frame_dir, frame_rate_extracted = extract_frames(temp_video_path, frame_interval_seconds)
            if saved_frames and frame_dir:
                st.session_state.saved_frames = saved_frames
                st.session_state.frame_dir = frame_dir
                st.session_state.frame_rate = frame_rate_extracted
                st.success(f"Frames extracted and saved every {frame_interval_seconds} seconds.")
            else:
                st.error("No frames were saved. Please check the video or interval.")

    # If frames have been extracted, proceed to segmentation
    if 'saved_frames' in st.session_state and st.session_state.saved_frames:
        frame_dir = st.session_state.frame_dir
        frame_names = sorted(
            [p for p in os.listdir(frame_dir) if p.endswith((".jpg", ".jpeg"))],
            key=lambda p: int(os.path.splitext(p)[0])
        )
        
        # Load the first frame
        first_frame_path = os.path.join(frame_dir, frame_names[0])
        first_frame = Image.open(first_frame_path).convert("RGB")

        st.write("Use the canvas below to select object points. For best results:")
        # st.write("- Choose the 'circle' tool from the toolbar on the left.")
        st.write("- Click on the image to place a small circle at the object location(s).")
        st.write("- Add as many object points as needed. Each circle represents one object.")

        # 1) Decide how big we allow the displayed image to be
        MAX_WIDTH = 800
        original_width, original_height = first_frame.size

        if original_width > MAX_WIDTH:
            aspect_ratio = original_height / original_width
            new_width = MAX_WIDTH
            new_height = int(MAX_WIDTH * aspect_ratio)
            first_frame = first_frame.resize((new_width, new_height), Image.ANTIALIAS)
        else:
            new_width = original_width
            new_height = original_height

        # 2) Compute the scaling factors (original / new)
        scale_factor_w = original_width / new_width
        scale_factor_h = original_height / new_height

        # 3) Create the canvas with the resized image
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            background_image=first_frame,
            update_streamlit=True,
            width=new_width,
            height=new_height,
            drawing_mode="circle",
            key="canvas"
        )

        if st.button("Segment"):
            # 4) Convert the scaled circles to original coordinates
            object_data = extract_points_from_canvas(
                canvas_result.json_data if canvas_result else {},
                scale_factor_w,
                scale_factor_h
            )
            if not object_data:
                st.error("No points selected. Please place at least one circle on the image.")
            else:
                st.write("Points selected (original coordinates):", object_data)

                # Initialize the model
                config_path = "sam2_configs"
                config_name = "sam2_hiera_l"
                sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"

                config_file = os.path.join(config_path, f"{config_name}.yaml")
                if not os.path.exists(config_file):
                    st.error(f"Configuration file not found at {config_file}. Please check the path.")
                    st.stop()

                if not os.path.exists(sam2_checkpoint):
                    st.error(f"Checkpoint file not found at {sam2_checkpoint}. Please check the path.")
                    st.stop()

                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                    st.warning("CUDA device not found. Running on CPU may be slow.")

                # Get the predictor (cached)
                try:
                    predictor = get_predictor(
                        config_path=config_path,
                        config_name=config_name,
                        sam2_checkpoint=sam2_checkpoint,
                        device=device
                    )
                except Exception as e:
                    st.error(f"Failed to initialize the predictor: {e}")
                    st.stop()

                # Run segmentation on all frames
                with st.spinner("Running segmentation on all frames..."):
                    try:
                        inference_state = predictor.init_state(video_path=frame_dir)
                        predictor.reset_state(inference_state)
                    except Exception as e:
                        st.error(f"Error initializing inference state: {e}")
                        st.stop()

                    # Add the user-drawn points to the first frame
                    try:
                        for idx, (x, y, label) in enumerate(object_data):
                            points = np.array([[x, y]], dtype=np.float32)
                            labels = np.array([label], dtype=np.int32)
                            predictor.add_new_points(
                                inference_state,
                                frame_idx=0,
                                obj_id=idx + 1,
                                points=points,
                                labels=labels
                            )
                    except Exception as e:
                        st.error(f"Error adding new points: {e}")
                        st.stop()

                    combined_frames = {}
                    try:
                        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                            frame_path = os.path.join(frame_dir, frame_names[out_frame_idx])
                            frame = Image.open(frame_path).convert('RGBA')
                            if out_frame_idx not in combined_frames:
                                combined_frames[out_frame_idx] = frame

                            for i, obj_id in enumerate(out_obj_ids):
                                mask = (out_mask_logits[i] > 0).cpu().numpy().squeeze()
                                mask_result = show_mask(mask, combined_frames[out_frame_idx])
                                combined_frames[out_frame_idx] = mask_result
                    except Exception as e:
                        st.error(f"Error during video propagation: {e}")
                        st.stop()

                    # Save segmented RGB results
                    output_dir = "RGB masks"
                    os.makedirs(output_dir, exist_ok=True)
                    try:
                        for frame_idx, combined_frame in combined_frames.items():
                            combined_frame.convert('RGB').save(
                                os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
                            )
                    except Exception as e:
                        st.error(f"Error saving segmented frames: {e}")
                        st.stop()

                st.success("Segmentation completed for all frames.")

                # Convert RGB masks to black and white
                input_dir = output_dir
                output_dir_bw = "Binary masks"
                os.makedirs(output_dir_bw, exist_ok=True)

                with st.spinner("Converting to binary masks..."):
                    try:
                        for filename in os.listdir(input_dir):
                            if filename.endswith(".png"):
                                img_path = os.path.join(input_dir, filename)
                                img = cv2.imread(img_path)
                                if img is not None:
                                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
                                    kernel = np.ones((3, 3), np.uint8)
                                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
                                    output_path = os.path.join(output_dir_bw, filename)
                                    cv2.imwrite(output_path, mask)
                    except Exception as e:
                        st.error(f"Error converting to binary masks: {e}")
                        st.stop()

                st.success("Binary masks saved.")

                # Calculate activity index
                with st.spinner("Calculating activity index..."):
                    try:
                        activity_values = calculate_activity_index(output_dir_bw)
                        fig = plot_activity_index(activity_values, frame_interval_seconds, video_duration)
                        plot_file_path = "activity_index_plot.png"
                        fig.savefig(plot_file_path, dpi=300)
                        st.success(f"Activity index plot saved to {plot_file_path} with 300 dpi resolution.")
                        st.pyplot(fig)
                        
                        activity_file_path = "activity_index.txt"
                        with open(activity_file_path, "w") as f:
                            f.write("Frame Number\tNormalized Activity Index\n")
                            for i, value in enumerate(activity_values):
                                f.write(f"{i+1}-{i+2}\t{value:.6f}\n")
                        st.success(f"Normalized activity index values saved to {activity_file_path}.")
                    except Exception as e:
                        st.error(f"Error calculating activity index: {e}")
                        st.stop()

                st.write("Activity index calculation and plotting completed!")

                # Optionally show difference images for first 5 pairs
                st.write("Below are the first 5 frame pairs with their difference:")
                for i in range(5):
                    frame1_path = os.path.join(output_dir_bw, f"frame_{i:05d}.png")
                    frame2_path = os.path.join(output_dir_bw, f"frame_{i+1:05d}.png")

                    frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
                    frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

                    if frame1 is None or frame2 is None:
                        st.warning(f"Skipping Frame {i} and Frame {i+1} due to missing files.")
                        continue

                    frame_diff = cv2.absdiff(frame1, frame2)

                    diff_fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(frame1, cmap='gray')
                    axes[0].set_title(f'Frame {i}')
                    axes[0].axis('off')

                    axes[1].imshow(frame2, cmap='gray')
                    axes[1].set_title(f'Frame {i+1}')
                    axes[1].axis('off')

                    axes[2].imshow(frame_diff, cmap='gray')
                    axes[2].set_title('Difference')
                    axes[2].axis('off')

                    st.pyplot(diff_fig)
    else:
        st.info("Please upload a video file to begin.")

if __name__ == "__main__":
    main()
