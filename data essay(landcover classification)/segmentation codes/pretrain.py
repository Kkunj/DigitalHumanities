import torch
import torchvision.models as models
import argparse
import os

def load_pretrained_model(args):
    if args.backbone == 'resnet50':
        net = models.resnet50(pretrained=False)
        net.fc = torch.nn.Linear(2048, 19)
    elif args.backbone == 'resnet18':
        net = models.resnet18(pretrained=False)
        net.fc = torch.nn.Linear(512, 19)

    if args.bands == 'all':
        net.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> Loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # Rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                del state_dict[k]

            msg = net.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> Loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> No checkpoint found at '{}'".format(args.pretrained))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for loading a pre-trained model and fine-tuning it.")
    parser.add_argument("--backbone", type=str, default='resnet18', choices=['resnet18', 'resnet50'], help="Backbone architecture")
    parser.add_argument("--bands", type=str, default='all', choices=['all'], help="Number of input bands")
    parser.add_argument("--pretrained", type=str, default='', help="Path to the pre-trained model file")

    args = parser.parse_args()
    load_pretrained_model(args)
