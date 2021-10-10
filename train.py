import argparse
import torchvision.models as models
from networks import *
from utils import *
from dog_breed_classifier import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--data_path", type=str, default="data/dogImages", help="Path to data"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs to train the model for",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--net",
        type=str,
        default="vgg16_tf",
        help="Select the network to use",
        choices=["scratch, vgg16_tf, resnet50_tf"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {}".format(device))

    # Model
    if args.net == "scratch":
        model = SimpleNet()
    elif args.net == "vgg16_tf":
        vgg16 = models.vgg16(pretrained=True)
        model = VGGTransferLearningNet(vgg16)
    elif args.net == "resnet50_tf":
        resnet50 = models.resnet50(pretrained=True)
        model = ResNetTransferLearningNet(resnet50)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup pipeline
    dog_breed_classifier_pipeline = DogBreedClassifierPipeline(
        data_path=args.data_path, model=model, save_prefix=args.net
    )

    dog_breed_classifier_pipeline.train(
        num_epochs=args.num_epochs, loss_fn=loss_fn, optimizer=optimizer, device=device
    )
