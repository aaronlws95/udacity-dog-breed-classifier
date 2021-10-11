import cv2
import torch
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from pathlib import Path
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_dog_dataset(path, one_hot_encoding=False):
    """
    Load the dog dataset
    Input:
        path (string): Path to dog dataset
    Output:
        dog_files: List of paths to dog images
        dog_targets: Corresponding dog one hot encoding target
        dog_targets_map: Mapping between dog name and target index
    """
    dog_files = [x for x in Path(path).rglob("**/*") if x.is_file()]
    dog_targets = [x.name for x in sorted(Path(path).glob("**/*")) if x.is_dir()]
    dog_targets_map = {x: i for i, x in enumerate(dog_targets)}
    num_classes = len(dog_targets_map.keys())

    if one_hot_encoding:
        dog_targets = [
            np.eye(num_classes, dtype="uint8")[i]
            for i in [dog_targets_map[x.parent.name] for x in dog_files]
        ]
    else:
        dog_targets = [dog_targets_map[x.parent.name] for x in dog_files]

    dog_targets_map = {i: x[4:] for x, i in dog_targets_map.items()}

    return dog_files, dog_targets, dog_targets_map


def load_human_dataset(path):
    """
    Load the human dataset
    Input:
        path (string): Path to dataset
    Output:
        human_files: List of paths to human images
    """
    human_files = [x for x in Path(path).rglob("**/*") if x.is_file()]
    return human_files


def get_num_faces_haarcascade_PIL_img(img, detector):
    """
    Use Haar Cascade to get number of faces in an image
    Input:
        img: RGB PIL image
        detector: Haar Cascade detector
    Output:
        Number of faces present in the image
    """
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray)
    return len(faces)


def get_num_faces_haarcascade(img_path, detector):
    """
    Use Haar Cascade to get number of faces in an image
    Input:
        img_path: Path to input image
        detector: Haar Cascade detector
    Output:
        Number of faces present in the image
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray)
    return len(faces)


def visualize_faces_haarcascade(img_path, detector):
    """
    Visualize faces obtained from Haar Cascade
    Input:
        img_path: Path to input image
        detector: Haar Cascade detector
    """
    # Load color (BGR) image
    img = cv2.imread(img_path)
    # Convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find faces in image
    faces = detector.detectMultiScale(gray)
    # Get bounding box for each detected face
    for (x, y, w, h) in faces:
        # add bounding box to color image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Display the image, along with bounding box
    plt.imshow(cv_rgb)
    plt.show()


def get_num_faces_mtcnn(img_path, detector):
    """
    Get number of faces in an image with MTCNN
    Input:
        img_path: Path to input image
        detector: MTCNN model
    Output:
        Number of faces present in the image
    """
    img = cv2.imread(img_path)
    faces = detector(img)
    return faces.shape[0] if faces is not None else 0


def visualize_faces_mtcnn(img_path, detector):
    """
    Visualize faces obtained from MTCNN
    Input:
        img_path: Path to input image
        detector: MTCNN model
    """
    img = cv2.imread(img_path)
    faces = detector(img)
    for x in faces:
        x = np.asarray(x)
        x = np.transpose(x, (1, 2, 0))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.normalize(
            x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        ).astype(np.uint8)
        plt.imshow(x)
        plt.show()


def PIL_to_tensor_imagenet(img, device):
    """
    Convert PIL image to PyTorch tensor with ImageNet preprocessing
    Input:
        img: RGB PIL image
        device: Name of device to use e.g. "cuda: 0" for GPU or "cpu"
    Output:
        img: PyTorch tensor with ImageNet preprocessing
    """
    transform = transforms.Compose(
        [
            transforms.Resize(size=(244, 244)),  # Resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = transform(img).unsqueeze(0).to(device)
    return img


def detect_dog_imagenet_PIL_img(img, model, device):
    """ "
    Classify if the given image is that of a dog
    Input:
        img: RGB PIL image
        model: ML ImageNet model to classify with
        device: Name of device to use e.g. "cuda: 0" for GPU or "cpu"
    """
    img = PIL_to_tensor_imagenet(img, device)
    out = model(img)
    idx = torch.max(out, 1)[1].item()
    return idx >= 151 and idx <= 268


def detect_dog_imagenet(img_path, model, device):
    """ "
    Classify if the given image is that of a dog
    Input:
        img_path (string): Path to image
        model: ML ImageNet model to classify with
        device: Name of device to use e.g. "cuda: 0" for GPU or "cpu"
    """
    img = Image.open(img_path).convert("RGB")
    return detect_dog_imagenet_PIL_img(img, model, device)
