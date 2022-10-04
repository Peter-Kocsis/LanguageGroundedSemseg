from torch import nn

from models.clip_models import Res16UNet34D
from models.modules.common import conv, conv_tr
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine import MinkowskiReLU

class Res16UNet34D_I(Res16UNet34D):
    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super().__init__(in_channels, out_channels, config, **kwargs)

        # Create a classification head
        head = conv(768, out_channels, kernel_size=1, stride=1, bias=True, D=D)
        # head = nn.Sequential(
        #     # conv(512, 64, kernel_size=1, stride=1, bias=True, D=D),
        #     # MinkowskiReLU(inplace=True),
        #     # conv(64, 64, kernel_size=1, stride=1, bias=True, D=D),
        #     # MinkowskiReLU(inplace=True),
        #     conv(64, out_channels, kernel_size=1, stride=1, bias=True, D=D),
        # )

        intermediate_layers = nn.ModuleDict({
            # 'p16p4': conv_tr(256, 64, kernel_size=4, upsample_stride=4, bias=True, D=D),
            # 'p4p1': conv_tr(384, 256, kernel_size=4, upsample_stride=4, bias=True, D=D),

            'p16p8': conv_tr(256, 128, kernel_size=2, upsample_stride=2, bias=True, D=D),
            'p8p4': conv_tr(384, 128, kernel_size=2, upsample_stride=2, bias=True, D=D),
            'p4p2': conv_tr(384, 128, kernel_size=2, upsample_stride=2, bias=True, D=D),
            'p2p1': conv_tr(384, 256, kernel_size=2, upsample_stride=2, bias=True, D=D),
        })
        self.final = nn.ModuleDict({
            'intermediate': intermediate_layers,
            'head': head
        })

    def forward(self, x):  # remove final classifier layer
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_b4p16 = self.block4(out)

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out_b4p16)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = me.cat(out, out_b3p8)
        out_b5p8 = self.block5(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out_b5p8)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = me.cat(out, out_b2p4)
        out_b6p4 = self.block6(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out_b6p4)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = me.cat(out, out_b1p2)
        out_b7p2 = self.block7(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out_b7p2)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = me.cat(out, out_p1)
        out = self.block8(out)

        if self.repr_only:
            return out
        else:
            # Concat the intermediate features
            # interm_b1 = self.intermediate_layers.b1p2(out_b1p2)
            # interm_b2 = self.intermediate_layers.b2p4(out_b2p4)
            # interm_b3 = self.intermediate_layers.b3p8(out_b3p8)
            # interm_b4 = self.intermediate_layers.b4p16(out_b4p16)
            #
            # interm_b4 = me.cat(out_b3p8, interm_b4, out_b5p8)
            # interm_b5 = self.intermediate_layers.b5p8(interm_b4)
            #
            # interm_b6 = self.intermediate_layers.b6p4(out_b6p4)
            # interm_b7 = self.intermediate_layers.b7p2(out_b7p2)
            # interm_b8 = self.intermediate_layers.b8(out)

            # interm = me.cat(interm_b1,
            #                 interm_b2,
            #                 interm_b3,
            #                 interm_b4,
            #                 interm_b5,
            #                 interm_b6,
            #                 interm_b7,
            #                 interm_b8)

            # interm_b4 = self.final.intermediate.p16p4(out_b4p16)
            # interm_p4 = me.cat(out_b2p4,
            #                    interm_b4,
            #                    out_b6p4)
            # interm_b2b6 = self.final.intermediate.p4p1(interm_p4)
            # interm_p1 = me.cat(interm_b2b6, out)

            interm_b4 = self.final.intermediate.p16p8(out_b4p16)
            interm_p8 = me.cat(interm_b4,
                               out_b5p8)

            interm_b5 = self.final.intermediate.p8p4(interm_p8)
            interm_p4 = me.cat(interm_b5,
                               out_b6p4)

            interm_b6 = self.final.intermediate.p4p2(interm_p4)
            interm_p2 = me.cat(interm_b6,
                               out_b7p2)

            interm_b7 = self.final.intermediate.p2p1(interm_p2)
            interm = me.cat(interm_b7,
                            out)

            return self.final.head(interm), out
            # return self.final.head(interm_p1), out