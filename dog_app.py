import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

class DogApp:
    def __init__(self, dog_breed_classifier_fn, dog_detector_fn, human_detector_fn):
        self.dog_breed_classifier_fn = dog_breed_classifier_fn
        self.dog_detector_fn = dog_detector_fn
        self.human_detector_fn = human_detector_fn

    def run(self, img_path, show_img=True):
        is_human = self.human_detector_fn(img_path)
        is_dog = self.dog_detector_fn(img_path)
        _, breed = self.dog_breed_classifier_fn(img_path)

        if show_img:
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.show()

        if is_human and is_dog:
            print("Can't tell if human or dog. Either way looks like a {}".format(breed))
        elif is_dog: 
            print("This is a {} dog".format(breed))
        elif is_human:
            print("Not a dog but looks like a {}".format(breed))
        else:
            print("Neither dog nor human")