# AnimalAI
An Open-Source Web Platform for Automated Animal Activity Index Calculation Using Interactive Deep-Learning Segmentation

## Installation

1. Clone the Repository:
   
```bash
git clone https://github.com/yourusername/VideoIntelligence.git
cd VideoIntelligence

2. Set Up the Environment:

conda create -n Myenv 
conda activate Myenv
pip install -r requirements.txt

3. Download the Model:

The file `checkpoints/sam2_hiera_large.pt` is too large to be hosted in this repository.  
Please download it from the following link and place it in the `checkpoints/` folder of this project:

https://github.com/facebookresearch/sam2 

Run the app with:

streamlit run app_new.py

How to Use the App:

Launch the App:
After running app_new.py, the Streamlit interface will open in your browser.

Upload a Video:
Click on the video uploader and select your video file (MP4, MOV, AVI).
The app will display the video duration and recommend a frame extraction interval.
Set the Frame Interval:

Input the desired frame interval and click "Extract Frames".
The app extracts frames from your video based on the interval.
Annotate the First Frame:

The first frame is displayed on an interactive canvas.
Use the circle tool to mark object points.
Segment the Video:

Click the "Segment" button.
The SAM2 model is initialized and segmentation is performed across all frames.
Segmented frames are saved and converted into binary masks.
View Activity Analysis:

The app calculates a normalized activity index based on frame differences.
A plot showing the activity index over time is displayed.
Optionally, difference images between consecutive frame pairs are shown.
