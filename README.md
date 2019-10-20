# AUTOMAP
This is an implementation of AUTOMAP which reconstructs MRI-images with undersampled k-space data

# Setup
1. Clone the repository
2. Create a virtual environment with python 3.6 (3.7 does not work)
3. Install the following packages:
- numpy 1.14.5
- scikit-learn (sklearn) 0.21.3
- tensorflow 1.7.1 (tensorflow 2 does not work)
- pillow 1.1.7

# Manual
1. preprocess_images\
This module preprocesses raw images in the images_raw directory. Processed images are stored in the root directory.

2. neural_network_train_and_save\
This module trains the neural network, undersamples the data with undersampling masks in the "pattern"-directory, creates a directory named "saved_models" and saves the network in this directory. 

3. neural_network_use_trained_model\
This module uses a trained module and saves the output images from given inputs.
A meta-file is created during the training process located in "saved_models". The name of this meta-file needs to be copied to the variable "saved_model_meta_info". This information needs to be set to load the correct model.
The reconstructed images are stored in folder that is created and named "recon".

Options:\
Change the target width and height from n=64 to the desired resolution. Higher resolutions might cause memory issues.
Use different training images by replacing the folders in images_raw.\
e.g. images_raw/custom_folder/custom_images (images must be in folder inside images_raw).
