# Video_SemanticSegmentation_MaskRCNN

Understanding Mask R-CNN

Mask R-CNN is an extension of Faster R-CNN that introduces a branch for pixel-wise object segmentation. It enhances previous methods by incorporating:

- Bounding Box Detection: Identifies objects in an image.
- Instance Segmentation: Predicts pixel-wise masks for each object.
- Feature Pyramid Networks (FPN): Improves multi-scale feature detection.
- ROI Align: Eliminates misalignment issues in object detection.
The original paper:
He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). "Mask R-CNN."


Repository Features

This project is an adapted version of alsombra/Mask_RCNN-TF2 with key enhancements:

- Upgraded to TensorFlow 2.x for modern compatibility.
- Pretrained COCO weights for transfer learning.
- Custom dataset support for instance segmentation.
- ROI Align implementation for precise mask extraction.
- Multi-GPU training support.
- Background removal & visualization tools.


Project Workflow

Step 1: Downloading the Repository
- Clone the repository to your local machine or Google Colab environment.

Step 2: Installing Required Dependencies
- Navigate to the repository folder (Mask_RCNN-TF2) and install all required libraries.
- Some dependencies require compatibility adjustments for TensorFlow 2.x.

 
Step 3: Importing the Required Libraries
  - 3(A): Installing Older Versions of NumPy
    I had to downgrade and install the numpy version (1.23.5) that:
      - supports 'np.bool' alias
      - is compatible to the Tensoflow version (2.15.0) to be installed below.
      - is compatible with Python version 3.11 I am currently using
  - 3(B): Installing Older Versions of TensorFlow
    - TensorFlow 2.16+ introduced changes that break compatibility with older .h5 weights. TensorFlow 2.15.0 still supports by_name=True for .h5 models without issues.
   
Step 4: Importing Files from the MRCNN Folder
- Import the necessary files from the MRCNN directory, which includes model architecture, utility, and visualisation functions.

Step 5: Importing the (coco) Dataset
  - 5(A): Creating the Logs and Accessing the Images Folder
    - Create a logs/ directory to store model checkpoints and training logs.
    - Access the images/ folder to load sample images for segmentation.
   
Step 6: Updating Compatibility with TensorFlow v2
- Since the original implementation was designed for TensorFlow 1.x, certain modifications are necessary. Make sure to import TensorFlow 1.x configuration and session classes while using the 2.x-compatible interface.

Step 7: Loading the Pre-trained Neural Network
  - 7(A): Specifying the Path and Data Format
    - Define the file path where the COCO pre-trained weights are stored.
  - 7(B): Downloading the Pre-trained Model and Weights
    - The model uses pre-trained COCO weights to perform instance segmentation on images.
    - These weights allow the model to detect multiple object classes.
  - 7(C): Specifying GPU Resources
    - Configure the GPU settings to optimize memory usage.
    - Ensure that TensorFlow allocates only necessary resources instead of consuming all available memory.
  - 7(D): Instantiating the Mask R-CNN Model for Inference
    - Create an instance of the MaskRCNN class in inference mode.
    - Set up the configuration parameters needed for testing.
  - 7(E): Loading the Weights into the Network
    - Load the pre-trained weights into the Mask R-CNN model.
    - Ensure that the weight format is compatible with the TensorFlow version being used.
   
Step 8: Segmenting in Videos
  - 8(A): Create a List of Objects from the COCO Dataset
    - Define a list of object classes that the model should detect in videos.
  - 8(B): Import .mp4 Video from Kaggle
    - Load a video file from Kaggle datasets for segmentation.
  - 8(C): Load the .mp4 File
    - Read the video file using OpenCV (cv2.VideoCapture()).
    - Extract frames for processing.
  - 8(D): Saving the Segmented Video
    - Define the format of saving the processed video
    - Save the segmented output using OpenCV's cv2.VideoWriter().
  - 8(E): Importing a Custom Function for Visualizing Segmented Videos
    - A custom file is provided to visualizing segmentation results, particularly for Mask R-CNN outputs, named 'video_functions.py'
    - Download the function from my 'Github' link
  - 8(F): Import the video_functions.py File to Project Session
    - Import video_functions.py from GitHub.
    - This file contains custom utilities for handling video segmentation.

Step 9: Visualizing the Objects in the Video
  - 9(A): Define a Function to Display an Image
    - Create a function to display segmented frames using Matplotlib or OpenCV.
  - 9(B): Set Processing Frequency and Customize Video Length
    - Allow users to define how many processed frames are displayed and lenght of video to process
  - 9(C): OpenCV-Based Custom Video Frame Processing with Time Limit
    - Implement frame-skipping logic to improve efficiency.
    - Limit video processing time to a user-defined duration.
  - 9(D): Display the Segmented .mp4 File
    - Use OpenCV (cv2.imshow()) or Colab's display(Video()) to visualize the output.

Conclusion

This repository provides a TensorFlow 2.x-compatible implementation of Mask R-CNN for instance segmentation.
It builds upon the original research paper while ensuring compatibility with modern libraries. Key features include:

- Accurate object detection and segmentation.
- Pretrained model inference using COCO weights.
