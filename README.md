# Dog Breed Classifier

## Project Overview
In this project, we develop an algorithm that takes in an image and identifies if contains a human or a dog. If it does contain either a human or a dog, a convolutional neural network (CNN) will classify the dog's breed or the resembling dog breed for that human. For our project we use the [Stanford Dogs dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) to train our dog breed classifier. Additionally, we develop a web application that allows users to upload their own images to be classified by our algorithm. 

This work was carried out as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

![Web app screenshot](/media/webapp_screenshot.png?raw=true)

## Setup
```
conda create -n udacity-dogenv python=3.7 #optional to use Miniconda
pip install -r requirements.txt
```

## Instructions
1. Download the datasets
    * [Dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
    * [Human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)
    * Ensure the files are saved according the file directory

2. Training a model (optional)
    * Models get saved to `pretrained/models`
    * Alternatively, you can use a pretrained model
```
python train.py --net scratch # train a simple model from scratch
python train.py --net resnet50_tf # train a ResNet50 model with transfer learning
python train.py --net vgg16_tf # train a VGG16 model with transfer learning
```

3. Running the web application
```
python run.py --data_path data/dogImages --net resnet50_tf --model_path pretrained/saved_models/resnet50_tf_2021_10_11_180421188539_06300_best.pt
```

4. Go to http://0.0.0.0:3001/ and upload an image of your choice!

## Files
```
dog-breed-classifier
├── data/                   # Store datasets
│   ├── dogImages/          # Dog dataset
|   │   ├── train/          # Training set
|   │   ├── valid/          # Validation set
|   │   └── test/           # Testing set
│   └── lfw/                # Human dataset
├── media/                  # Store images
├── pretrained/             # Store pretrained data
│   ├── haarcascades/       # Pretrained data for the Haar Cascade detector
│   └── saved_models/       # Pretrained machine learning models
├── static/images/          # Static images for the web application
├── templates
│   ├── go.html             # Handle requests for the web application
│   └── master.html         # Main index for the web application
├── dog_app.ipynb           # Exploratory notebook
├── dog_app.html            # Exploratory saved to HTML
├── dog_app.py              # Full application algorithm
├── dog_breed_classifier.py # Dog breed classification pipeline
├── networks.py             # Machine learning network definitions
├── train.py                # Machine learning training script
├── utils.py                # Utility functions
├── run.py                  # Flask web application script
├── requirements.txt        # Pip requirements
└── README.md               # Documentation
```

## Pretrained models
* [VGG16 Transfer Learning](https://drive.google.com/file/d/1V95M1Aaz9Vd_BDO7pKYavwBRt7HB5dt7/view?usp=sharing)

## Resources
* [Labeled Faces in the Wild dataset](http://vis-www.cs.umass.edu/lfw/): [download link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)
* [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/): [download link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* [OpenCV Haar feature-based cascade classsifier](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)
    * [Pretrained detector](https://github.com/opencv/opencv/tree/master/data/haarcascades)
* [three_people.jpg](https://www.shutterstock.com/image-photo/horizontal-shot-three-mixed-race-teenagers-1238409808)
* [Face recognition using PyTorch](https://github.com/timesler/facenet-pytorch)