import matplotlib.pyplot as plt
from PIL import Image


class DogApp:
    """
    Application that takes in an image and detects if there is a dog or human in the image.
    If there is either one, the application then classifies the breed of the dog or human.
    """

    def __init__(self, dog_breed_classifier_fn, dog_detector_fn, human_detector_fn):
        """
        Initialize the application
        Input:
            dog_breed_classifier_fn (x): Dog breed classification lambda function that takes in either an image
                                         or image path and outputs the breed of the dog
            dog_detector_fn (x): Dog detector lambda function that takes in either an image or an image path and
                                outputs True if there is a dog in the image
            human_detector_fn (x): Human detector lambda function that takes in either an image or an image path
                                    and outputs True if there is at least one human in the image
        """
        self.dog_breed_classifier_fn = dog_breed_classifier_fn
        self.dog_detector_fn = dog_detector_fn
        self.human_detector_fn = human_detector_fn

    def run(self, img):
        """
        Run the app
        Input:
            img: Either an RGB PIL image or image path depending on the initialized lambda functions
        Output:
            The results of the application given as a string
        """
        is_human = self.human_detector_fn(img)
        is_dog = self.dog_detector_fn(img)

        if not is_human and not is_dog:
            return "Neither dog nor human"

        _, breed = self.dog_breed_classifier_fn(img)

        if is_human and is_dog:
            return "Can't tell if human or dog. Either way looks like a {}".format(
                breed.replace("_", " ")
            )
        elif is_dog:
            return "This is a {} dog".format(breed.replace("_", " "))
        elif is_human:
            return "Not a dog but looks like a {}".format(breed.replace("_", " "))
