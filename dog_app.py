import matplotlib.pyplot as plt
from PIL import Image


class DogApp:
    def __init__(self, dog_breed_classifier_fn, dog_detector_fn, human_detector_fn):
        self.dog_breed_classifier_fn = dog_breed_classifier_fn
        self.dog_detector_fn = dog_detector_fn
        self.human_detector_fn = human_detector_fn

    def run(self, img):
        is_human = self.human_detector_fn(img)
        is_dog = self.dog_detector_fn(img)
        _, breed = self.dog_breed_classifier_fn(img)

        if is_human and is_dog:
            return "Can't tell if human or dog. Either way looks like a {}".format(
                breed.replace("_", " ")
            )
        elif is_dog:
            return "This is a {} dog".format(breed.replace("_", " "))
        elif is_human:
            return "Not a dog but looks like a {}".format(breed.replace("_", " "))
        else:
            return "Neither dog nor human"
