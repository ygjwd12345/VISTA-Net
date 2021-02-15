##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Guanglei Yang
## Department of Information Engineering and Computer Science, University of Trento
## Email: guanglei.yang@studenti.unitn.it or yangguanglei.phd@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.functional import unfold
from encoding.nn import BatchNorm2d
from encoding.utils.non_local_embedded_gaussian import NONLocalBlock2D


def att_map(x):
    ### sptial attention
    a = torch.sum(x ** 2, dim=1)
    ### channel attention
    for i in range(a.shape[0]):
        a[i] = a[i] / torch.norm(a[i])
    a = torch.unsqueeze(a, 1)
    x = a.detach() * x
    return x
class multihead(nn.Module):
    def __init__(self,scale=3,heads=1,channel=256,norm_layer=BatchNorm2d):
        super(multihead, self).__init__()
        self.scale=scale
        if scale==3:
        ### confusion layer 3,4,5
            self.att_self_3 = nn.ModuleList(
                [NONLocalBlock2D(in_channels=2 * channel) for c in range(heads)]
            )
            self.att_self_4 = nn.ModuleList(
                [NONLocalBlock2D(in_channels=4 * channel) for c in range(heads)]
            )
            self.att_self_5 = nn.ModuleList(
                [NONLocalBlock2D(in_channels=8 * channel) for c in range(heads)]
            )
            self.confuslayer=nn.Conv2d((2+4+8)*channel, 8*channel, 1)
            self.deconvfinal=nn.Sequential(
                nn.Conv2d(8*channel, 8*channel,  kernel_size=3,stride=2,padding=1),
                norm_layer(8*channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(8 * channel, 8 * channel, kernel_size=3, stride=2, padding=1),
                norm_layer(8 * channel),
                nn.ReLU(inplace=True)
                                        )
        elif scale==4:
        ### confusion layer 2,3,4,5
            self.att_self_2 = nn.ModuleList(
                [NONLocalBlock2D(in_channels = channel) for c in range(heads)]
            )
            self.att_self_3 = nn.ModuleList(
                [NONLocalBlock2D(in_channels=2 * channel) for c in range(heads)]
            )
            self.att_self_4 = nn.ModuleList(
                [NONLocalBlock2D(in_channels=4 * channel) for c in range(heads)]
            )
            self.att_self_5 = nn.ModuleList(
                [NONLocalBlock2D(in_channels=8 * channel) for c in range(heads)]
            )
            self.confuslayer = nn.Conv2d((1 + 2 + 4 + 8) * channel, 8*channel, 1)
            self.deconvfinal = nn.Sequential(
                nn.Conv2d(8 * channel, 8 * channel, kernel_size=3, stride=2, padding=1),
                norm_layer(8 * channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(8 * channel, 8 * channel, kernel_size=3, stride=2, padding=1),
                norm_layer(8 * channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(8 * channel, 8 * channel, kernel_size=3, stride=2, padding=1),
                norm_layer(8 * channel),
                nn.ReLU(inplace=True)
            )

        elif scale ==2:
        ### confusion layer 4,5
            self.att_self_4 = nn.ModuleList(
                [NONLocalBlock2D(in_channels=4 * channel) for c in range(heads)]
            )
            self.att_self_5 = nn.ModuleList(
                [NONLocalBlock2D(in_channels=8 * channel) for c in range(heads)]
            )
            self.confuslayer = nn.Conv2d((4 + 8) * channel, 8*channel, 1)
            self.deconvfinal = nn.Sequential(
                nn.Conv2d(8 * channel, 8 * channel, kernel_size=3, stride=2, padding=1),
                norm_layer(8 * channel),
                nn.ReLU(inplace=True)
            )
        else:
            print('using single scale now!!!')
            self.att_self_5 = nn.ModuleList(
                [NONLocalBlock2D(in_channels=8 * channel) for c in range(heads)]
            )
            self.deconvfinal = nn.Sequential(
                nn.Conv2d(8 * channel, 8 * channel, kernel_size=3, stride=1, padding=1),
                norm_layer(8 * channel),
                nn.ReLU(inplace=True)
            )
        ### deconv layer
        self.decov3=nn.Sequential(
            nn.Conv2d(2*heads*channel, 2*channel, kernel_size=3,stride=1,padding=1),
            norm_layer(2*channel),
            nn.ReLU(inplace=True)
        )
        self.decov2=nn.Sequential(
            nn.Conv2d(heads*channel, channel,  kernel_size=3,stride=1,padding=1),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.decov4=nn.Sequential(
            nn.Conv2d(4*heads*channel, 4*channel,  kernel_size=3,stride=1,padding=1),
            norm_layer(4*channel),
            nn.ReLU(inplace=True)
        )
        self.decov5=nn.Sequential(
            nn.Conv2d(8*heads*channel, 8*channel,  kernel_size=3,stride=1,padding=1),
            norm_layer(8*channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, c2, c3, c4, c5):
        if self.scale==3:
            _, _, h, w = c3.size()
            c4 = F.upsample(c4, (h, w))
            c5 = F.upsample(c5, (h, w))
            out_3 = []
            for mod in self.att_self_3:
                out_3.append(mod(c3))
            att_3 = torch.cat(out_3, dim=1)
            att_mh_3=self.decov3(att_3)
            out_4 = []
            for mod in self.att_self_4:
                out_4.append(mod(c4))
            att_4 = torch.cat(out_4, dim=1)
            att_mh_4=self.decov4(att_4)
            out_5 = []
            for mod in self.att_self_5:
                out_5.append(mod(c5))
            att_5 = torch.cat(out_5, dim=1)
            att_mh_5=self.decov5(att_5)
            feat_confuse=torch.cat([att_mh_3,att_mh_4,att_mh_5], dim=1)
            feat_confuse =self.confuslayer(feat_confuse)
        elif self.scale ==4:
            _, _, h, w = c2.size()
            c3 = F.upsample(c3, (h, w))
            c4 = F.upsample(c4, (h, w))
            c5 = F.upsample(c5, (h, w))
            out_2 = []
            for mod in self.att_self_2:
                out_2.append(mod(c2))
            att_2 = torch.cat(out_2, dim=1)
            att_mh_2=self.decov2(att_2)
            out_3 = []
            for mod in self.att_self_3:
                out_3.append(mod(c3))
            att_3 = torch.cat(out_3, dim=1)
            att_mh_3=self.decov3(att_3)
            out_4 = []
            for mod in self.att_self_4:
                out_4.append(mod(c4))
            att_4 = torch.cat(out_4, dim=1)
            att_mh_4=self.decov4(att_4)
            out_5 = []
            for mod in self.att_self_5:
                out_5.append(mod(c5))
            att_5 = torch.cat(out_5, dim=1)
            att_mh_5=self.decov5(att_5)
            feat_confuse=torch.cat([att_mh_2,att_mh_3,att_mh_4,att_mh_5], dim=1)
            feat_confuse =self.confuslayer(feat_confuse)
        elif self.scale ==2:
            _, _, h, w = c4.size()
            c5 = F.upsample(c5, (h, w))
            out_4 = []
            for mod in self.att_self_4:
                out_4.append(mod(c4))
            att_4 = torch.cat(out_4, dim=1)
            att_mh_4=self.decov4(att_4)
            out_5 = []
            for mod in self.att_self_5:
                out_5.append(mod(c5))
            att_5 = torch.cat(out_5, dim=1)
            att_mh_5=self.decov5(att_5)
            feat_confuse=torch.cat([att_mh_4,att_mh_5], dim=1)
            feat_confuse =self.confuslayer(feat_confuse)
        else:
            out_5 = []
            for mod in self.att_self_5:
                out_5.append(mod(c5))
            att_5 = torch.cat(out_5, dim=1)
            att_mh_5 = self.decov5(att_5)
            feat_confuse=att_mh_5
        ### add self spatial attention
        att = att_map(feat_confuse)
        out = att*c5+c5
        out=self.deconvfinal(out)

        return out



