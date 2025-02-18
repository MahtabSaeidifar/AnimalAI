# AnimalAI: An Open-Source Web Platform for Automated Animal Activity Index Calculation Using Interactive Deep-Learning Segmentation!


---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MahtabSaeidifar/AnimalAI.git
cd AnimalAI
```

### 2. Set Up the Environment
We recommend using [conda](https://docs.conda.io/en/latest/) for environment management:

```bash
conda create -n Myenv
conda activate Myenv
pip install -r requirements.txt
```

### 3. Download the Model
The file **`checkpoints/sam2_hiera_large.pt`** is too large to be hosted in this repository.  
Please download it from the [facebookresearch/sam2](https://github.com/facebookresearch/sam2) repository and place it in the `checkpoints/` folder of this project:

```
AnimalAI/
└── checkpoints/
    └── sam2_hiera_large.pt
```

### 4. Run the App
Use the following command to launch the Streamlit application:

```bash
streamlit run app_new.py
```

---

## How to Use the App

### 1. Launch the App
After running `streamlit run app_new.py`, the Streamlit interface will open automatically in your web browser.

### 2. Upload a Video
1. Click on the **video uploader** and select your video file (`.mp4`, `.mov`, `.avi`).
2. The app will display the video duration and recommend a frame extraction interval.

### 3. Set the Frame Interval
1. Input your desired **frame interval**.
2. Click **Extract Frames** to create frames from your video based on the specified interval.

### 4. Annotate the First Frame
1. The first frame is displayed on an **interactive canvas**.
2. Use the **circle tool** to mark the object(s) you want to track or segment.

### 5. Segment the Video
1. Click the **Segment** button.
2. The **SAM2** model will initialize and perform segmentation across all frames.
3. Segmented frames will be saved and converted into binary masks.

### 6. View Activity Analysis
1. The app calculates a **normalized activity index** based on differences between frames.
2. A plot showing this activity index over time is displayed.
3. Optionally, **difference images** between consecutive frames can be viewed for deeper analysis.


