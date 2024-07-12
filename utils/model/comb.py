from utils.model.dircn.dircn import DIRCN
from utils.model.kbnet.kbnet_l import KBNet_l
from utils.model.kbnet.kbnet_s import KBNet_s
from utils.model.nafnet.nafnet import NAFNet
from utils.model.varnet.varnet import VarNet
import torch
import torch.nn as nn

class CombNet(nn.Module):
    def __init__(self, args):

        super().__init__()

        if args.first_net_name == 'dircn':
            self.first_net = DIRCN(num_cascades=args.cascade,
                                   n=args.chans,
                                   sense_n=args.sens_chans)
        else:
            self.first_net = VarNet(num_cascades=args.cascade,
                                    chans=args.chans,
                                    sens_chans=args.sens_chans)

        if args.second_net_name == 'kbnet_s':
            self.second_net = KBNet_s()
        elif args.second_net_name == 'kbnet_l':
            self.second_net = KBNet_l()
        else:
            self.second_net = NAFNet()

    def forward(self, image_input: torch.Tensor, image_grappa: torch.Tensor, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        image_recon = self.first_net(masked_kspace, mask)

        image_input = self._crop_image(image_input)
        image_grappa = self._crop_image(image_grappa)
        image_concat = torch.cat((image_input, image_recon, image_grappa), dim=1)
        result = self.second_net(image_concat)

        return result

    def _crop_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform center crop for given image to size (384, 384)
        """
        height = image.shape[-2]
        width = image.shape[-1]
        return image[..., (height - 384) // 2: 384 + (height - 384) // 2, (width - 384) // 2: 384 + (width - 384) // 2]