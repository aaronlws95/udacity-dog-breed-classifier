import torch
import argparse
import torchvision.models as models
import torch.nn as nn
from networks import SimpleNet, VGGTransferLearningNet, ResNetTransferLearningNet
from dog_breed_classifier import DogBreedClassifierPipeline


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
        choices=["scratch", "vgg16_tf", "resnet50_tf"],
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
        model = VGGTransferLearningNet(models.vgg16(pretrained=True))
    elif args.net == "resnet50_tf":
        model = ResNetTransferLearningNet(models.resnet50(pretrained=True))

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup pipeline
    dog_breed_classifier_pipeline = DogBreedClassifierPipeline(
        data_path=args.data_path, model=model, save_prefix=args.net
    )

    # Train
    dog_breed_classifier_pipeline.train(
        num_epochs=args.num_epochs, loss_fn=loss_fn, optimizer=optimizer, device=device
    )
