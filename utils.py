import cv2
import torch
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path


def load_dog_dataset(path, one_hot_encoding=False):
    """
    Input:
        path (string): Path to dog dataset
    Output:
        dog_files: Path to dog image
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

    return dog_files, dog_targets, dog_targets_map


def load_human_dataset(path):
    human_files = [x for x in Path(path).rglob("**/*") if x.is_file()]
    return human_files


def get_num_faces_haarcascade(img_path, detector):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray)
    return len(faces)


def visualize_faces_haarcascade(img_path, detector):
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
    img = cv2.imread(img_path)
    faces = detector(img)
    return faces.shape[0] if faces is not None else 0


def visualize_faces_mtcnn(img_path, detector):
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


def load_image_imagenet(img_path, device):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(size=(244, 244)),  # Resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = transform(img).unsqueeze(0).to(device)
    return img


def detect_dog_imagenet(img_path, model, device):
    img = load_image_imagenet(img_path, device)
    out = model(img)
    idx = torch.max(out, 1)[1].item()
    return idx >= 151 and idx <= 268
