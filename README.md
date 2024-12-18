RSCNN - AI Model for Image Reconstruction
This repository contains RSCNN, a neural network designed to enhance image quality through a reconstruction process. The model was trained on a dataset of 144,000 images with dimensions of 256x256 to produce sharp and detailed results after upscaling.

🧠 Model Workflow
Training Data Preparation:

Original images were downscaled and blurred.
The degraded images were then upscaled to their original size (256x256) using the bicubic algorithm.
These degraded images were used as inputs to train the RSCNN model, with the original images serving as references.
Prediction Process:

The model takes an input image, performs an initial upscale using bicubic, and then applies a prediction to generate an enhanced version of the image.
The upscale factor is configurable to suit your needs.
⚙️ Usage
Requirements
Python 3.x
Install the required dependencies with:
bash
Copier le code
pip install -r requirements.txt
Usage Guide
Configure the input image path:

Open the main.py file.
Update the input image path to point to your desired image.
Run the script:

Execute main.py:
bash
Copier le code
python main.py
The model's output will be saved to the file ./output.jpg.
Example Usage
Here’s an example modification in main.py:

python
Copier le code
input_path = "./path/to/your/image.jpg"  # Input image path
output_path = "./output.jpg"            # Output image path
📂 Repository Structure
graphql
Copier le code
.
├── main.py                 # Main script to use the model
├── model/                  # RSCNN model source code
├── data/                   # (Optional) Folder to store images
├── requirements.txt        # Python dependencies
└── README.md               # This file
🖼️ Sample Results
Input Image (Degraded)	RSCNN Output
