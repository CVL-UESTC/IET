import argparse
import os
import os.path as osp

import torch
from PIL import Image
from torchvision import transforms

from basicsr.archs.iet_arch import IET


model_path = {
    "classical": {
        "2": "experiments/pretrained_models/IET-SR-classical-x2.pth",
        "3": "experiments/pretrained_models/IET-SR-classical-x3.pth",
        "4": "experiments/pretrained_models/IET-SR-classical-x4.pth",
    },
    "lightweight": {
        "2": "experiments/pretrained_models/IET-SR-light-x2.pth",
        "3": "experiments/pretrained_models/IET-SR-light-x3.pth",
        "4": "experiments/pretrained_models/IET-SR-light-x4.pth",
    }
}

model_config = {
    "classical": {
        "nattn_dim": 240,
        "topk_focus": [[914, 914, 225, 225], [345, 225, 225, 125], [225, 125, 125, 125], [185, 81, 81, 81],
                       [121, 81, 81, 81], [64, 64, 64, 64], [36, 36, 36, 36], [24, 24, 24, 24]],
        "topk_prop_1": [22, 20, 14, 12, 0, 0, 0, 0],
        "topk_prop_2": [12, 11, 9, 8, 0, 0, 0, 0],
        "topk_prop_3": [120, 100, 60, 40, 0, 0, 0, 0],
        "in_chans": 3,
        "img_size": 82,
        "embed_dim": 240,
        "depths": [4, 4, 4, 4, 4, 4, 4, 4],
        "num_heads": 6,
        "local_range": 17,
        "sparse_range": 25,
        "dilation": 3,
        "convffn_kernel_size": 5,
        "img_range": 1.,
        "mlp_ratio": 2,
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv",
        "use_checkpoint": False,
    },
    "lightweight": {
        "nattn_dim": 60,
        "topk_focus": [[914, 225, 225], [345, 225, 125], [225, 125, 125], [185, 81, 81],
                       [121, 81, 81], [64, 64, 64], [36, 36, 36], [36, 36, 36]],
        "topk_prop_1": [22, 20, 14, 12, 0, 0, 0, 0],
        "topk_prop_2": [12, 11, 9, 8, 0, 0, 0, 0],
        "topk_prop_3": [120, 100, 60, 40, 0, 0, 0, 0],
        "in_chans": 3,
        "img_size": 78,
        "embed_dim": 54,
        "depths": [3, 3, 3, 3, 3, 3, 3, 3],
        "num_heads": 3,
        "local_range": 17,
        "sparse_range": 25,
        "dilation": 3,
        "convffn_kernel_size": 7,
        "img_range": 1.,
        "mlp_ratio": 1,
        "upsampler": "pixelshuffledirect",
        "resi_connection": "1conv",
        "use_checkpoint": False,
    },
}


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="datasets/TestDataSR/LR/LRBI/Urban100/x2/img_010.png", help="Input image or directory path.")
    parser.add_argument("-o", "--out_path", type=str, default="results/test/", help="Output directory path.")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor for SR.")
    parser.add_argument(
            "--task",
            type=str,
            default="lightweight",
            choices=['classical', 'lightweight'],
            help="Task for the model. classical: for classical SR models. lightweight: for lightweight models."
            )
    args = parser.parse_args()

    return args



def process_image(image_input_path, image_output_path, model, device):
    with torch.no_grad():
        image_input_ = Image.open(image_input_path).convert('RGB')
        image_input = transforms.ToTensor()(image_input_).unsqueeze(0).to(device)
        image_output = model(image_input)
        if isinstance(image_output, (tuple, list)):
            image_output = image_output[0]
        image_output = image_output.float().clamp(0.0, 1.0)[0].cpu()

        image_output = transforms.ToPILImage()(image_output)
        image_output.save(image_output_path)

def main():
    args = get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IET(upscale=args.scale, **model_config[args.task])

    checkpoint = torch.load(model_path[args.task][str(args.scale)], map_location=device)
    state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if os.path.isdir(args.in_path):
        for file in os.listdir(args.in_path):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                image_input_path = osp.join(args.in_path, file)
                file_name = osp.splitext(file)
                image_output_path = os.path.join(args.out_path, file_name[0] + '_IET_' + args.task + '_SRx' + str(args.scale) + file_name[1])
                process_image(image_input_path, image_output_path, model, device)
    else:
        if args.in_path.endswith('.png') or args.in_path.endswith('.jpg') or args.in_path.endswith('.jpeg'):
            image_input_path = args.in_path
            file_name = osp.splitext(osp.basename(args.in_path))
            image_output_path = os.path.join(args.out_path, file_name[0] + '_IET_' + args.task + '_SRx' + str(args.scale) + file_name[1])
            process_image(image_input_path, image_output_path, model, device)


if __name__ == "__main__":
    main()
