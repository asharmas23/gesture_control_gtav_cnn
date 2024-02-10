# Hand Gesture Recognition for Game Control

## Objective

This project aims to develop a system that allows users to control computer applications, such as video games, using hand gestures. By leveraging background subtraction techniques and a convolutional neural network (CNN), the system interprets hand gestures as inputs for control actions, offering an intuitive and interactive user experience.

## Installation

1. Clone this repository to your local machine.
    git clone https://github.com/asharmas23/gesture_control_gtav_cnn.git

2. Navigate to the project directory.
    cd gesture_control_gtav_cnn


## Usage

To use this project for hand gesture recognition:
1. Start the process by first collecting the dataset, training the CNN model & the testing it using the appropriate scripts.
2. Ensure your webcam is connected and properly configured.
3. Perform hand gestures within the webcam's view to interact with the game or application.

## Dependencies

- Python 3.x
- Keras
- OpenCV
- NumPy

## Methodology

### Making Dataset
- Frames are captured from the webcam using the OpenCV library, converted to grayscale, and noise is reduced with GaussianBlur.
- Background is estimated using the first 150 frames, and subsequent 800 frames are used for training data per class after background subtraction and image enhancement.
- The dataset consists of 6400 training images across 8 classes and 1600 test images.

### Model Architecture and Training
- A sequential CNN model is created using Keras, incorporating convolution layers, max pooling, dropout layers, and dense layers.
- The model architecture includes ELU activation functions, categorical cross-entropy loss, and the Adam optimizer.
- Data augmentation is applied using ImageDataGenerator to enhance the dataset.

### Testing and Application
- The trained model is loaded and used to interpret live webcam feed.
- Predictions from the model map hand gestures to corresponding control actions in the application.

## Contributing

Please feel free to contribute to this project. You can fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to everyone who contributed to the development and testing of this project.
- Inspired by the need for innovative human-computer interaction methods.

    
