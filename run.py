import torch
import cv2
import argparse
import torchvision.models as models
from flask import Flask, request, render_template, session
from flask_dropzone import Dropzone
from utils import detect_dog_imagenet, get_num_faces_haarcascade
from dog_breed_classifier import DogBreedClassifierPipeline
from networks import SimpleNet, ResNetTransferLearningNet, VGGTransferLearningNet
from dog_app import DogApp

# Parse arguments
parser = argparse.ArgumentParser(description="Run web app")
parser.add_argument(
    "--data_path", type=str, default="data/dogImages", help="Path to dog dataset"
)
parser.add_argument(
    "--net",
    type=str,
    default="resnet50_tf",
    help="Select the network to use",
    choices=["scratch", "vgg16_tf", "resnet50_tf"],
)
parser.add_argument(
    "--model_path",
    type=str,
    default="pretrained/saved_models/resnet50_tf_2021_10_11_180421188539_06300_best.pt",
    help="Path to saved model",
)
args = parser.parse_args()

# Flask setup
app = Flask(__name__)
dropzone = Dropzone(app)
app.config.update(
    DROPZONE_ALLOWED_FILE_TYPE="image",
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_UPLOAD_ON_CLICK=True,
)
app.secret_key = "dog_app"

# Dog app setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {}".format(device))
# Model
if args.net == "scratch":
    model = SimpleNet()
elif args.net == "vgg16_tf":
    model = VGGTransferLearningNet(models.vgg16(pretrained=True))
elif args.net == "resnet50_tf":
    model = ResNetTransferLearningNet(models.resnet50(pretrained=True))

dog_breed_classifier_pipeline = DogBreedClassifierPipeline(
    data_path=args.data_path, model=model
)
dog_breed_classifier_pipeline.load(args.model_path)
dog_breed_classifier_fn = lambda x: dog_breed_classifier_pipeline.classify(x, device)
vgg16 = models.vgg16(pretrained=True).to(device).eval()
dog_detector_fn = lambda x: detect_dog_imagenet(x, vgg16, device)
face_cascade = cv2.CascadeClassifier(
    "pretrained/haarcascades/haarcascade_frontalface_alt.xml"
)
human_detector_fn = lambda x: get_num_faces_haarcascade(x, face_cascade)
dog_app = DogApp(dog_breed_classifier_fn, dog_detector_fn, human_detector_fn)

# Index page
@app.route("/")
@app.route("/index")
def index():
    session["result"] = ""
    return render_template("master.html")


# Handle query and display results
@app.route("/go", methods=["POST", "GET"])
def go():
    if request.method == "POST":
        for key, f in request.files.items():
            if key.startswith("file"):
                img_path = "static/images/tmp_img." + f.filename.split(".")[-1]
                f.save(img_path)
                session["result"] = dog_app.run(img_path)
                session["img_path"] = img_path
                break

    return render_template("go.html", result=session["result"], img=session["img_path"])


def main():
    app.run(host="0.0.0.0", port=3001, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
