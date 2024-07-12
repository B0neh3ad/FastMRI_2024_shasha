import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_image_data_loaders
from utils.model.kbnet.kbnet_l import KBNet_l
from utils.model.kbnet.kbnet_s import KBNet_s
from utils.model.nafnet.nafnet import NAFNet

from tqdm import tqdm

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (image, _, _, fnames, slices) in tqdm(data_loader):
            image = image.cuda(non_blocking=True)
            output = model(image)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None

def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device: ', torch.cuda.current_device())

    # second model (image-to-image)
    if args.net_name == 'kbnet_s':
        model = KBNet_s()
    elif args.net_name == 'kbnet_l':
        model = KBNet_l()
    else:
        model = NAFNet()
    model.to(device=device)

    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print("checkpoint's epoch:", checkpoint['epoch'], "/ best validation loss:", checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])

    forward_loader = create_image_data_loaders(
        image_data_path=args.data_path,
        recon_path=args.recon_path,
        args=args,
        isforward=True
    )
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)