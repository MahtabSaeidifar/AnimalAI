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
