# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.RRDBNet_arch import RRDBNet
from networks.FlowUpsamplerNet import FlowUpsamplerNet
import networks.thops as thops
import networks.flow as flow
from utils.util import opt_get


class NCSRNet(nn.Module):
    def __init__(
        self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, opt=None, step=None
    ):
        super(NCSRNet, self).__init__()

        self.opt = opt
        self.quant = (
            255
            if opt_get(opt, ["datasets", "train", "quant"]) is None
            else opt_get(opt, ["datasets", "train", "quant"])
        )
        self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)
        hidden_channels = opt_get(opt, ["network_G", "flow", "hidden_channels"])
        hidden_channels = hidden_channels or 64
        self.RRDB_training = True  # Default is true

        train_RRDB_delay = opt_get(self.opt, ["network_G", "train_RRDB_delay"])
        set_RRDB_to_train = False
        if set_RRDB_to_train:
            self.set_rrdb_training(True)

        self.flowUpsamplerNet = FlowUpsamplerNet(
            (160, 160, 3),
            hidden_channels,
            K,
            flow_coupling=opt["network_G"]["flow"]["coupling"],
            opt=opt,
        )
        self.i = 0

    def set_rrdb_training(self, trainable):
        if self.RRDB_training != trainable:
            for p in self.RRDB.parameters():
                p.requires_grad = trainable
            self.RRDB_training = trainable
            return True
        return False

    def forward(
        self,
        gt=None,
        lr=None,
        z=None,
        eps_std=None,
        reverse=False,
        epses=None,
        reverse_with_grad=False,
        lr_enc=None,
        add_gt_noise=False,
        step=None,
        y_label=None,
        std=None,
    ):
        if not reverse:
            return self.normal_flow(
                gt,
                lr,
                epses=epses,
                lr_enc=lr_enc,
                add_gt_noise=add_gt_noise,
                step=step,
                y_onehot=y_label,
                std=std,
            )
        else:
            # assert lr.shape[0] == 1
            assert lr.shape[1] == 3

            if reverse_with_grad:
                return self.reverse_flow(
                    lr,
                    z,
                    y_onehot=y_label,
                    eps_std=eps_std,
                    epses=epses,
                    lr_enc=lr_enc,
                    add_gt_noise=add_gt_noise,
                    std=std,
                )
            else:
                with torch.no_grad():
                    return self.reverse_flow(
                        lr,
                        z,
                        y_onehot=y_label,
                        eps_std=eps_std,
                        epses=epses,
                        lr_enc=lr_enc,
                        add_gt_noise=add_gt_noise,
                        std=std,
                    )

    def normal_flow(
        self,
        gt,
        lr,
        y_onehot=None,
        epses=None,
        lr_enc=None,
        add_gt_noise=True,
        step=None,
        std=None,
    ):
        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt
        if add_gt_noise:
            # Setup
            noiseQuant = opt_get(
                self.opt, ["network_G", "flow", "augmentation", "noiseQuant"], True
            )
            if noiseQuant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(
            rrdbResults=lr_enc,
            gt=z,
            logdet=logdet,
            reverse=False,
            epses=epses,
            y_onehot=y_onehot,
            std=std,
        )

        objective = logdet.clone()

        if isinstance(epses, (list, tuple)):
            z = epses[-1]
        else:
            z = epses

        objective = objective + flow.GaussianDiag.logp(None, None, z)

        nll = (-objective) / float(np.log(2.0) * pixels)

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)
        block_idxs = (
            opt_get(self.opt, ["network_G", "flow", "stackRRDB", "blocks"]) or []
        )
        if len(block_idxs) > 0:
            concat = torch.cat(
                [rrdbResults["block_{}".format(idx)] for idx in block_idxs], dim=1
            )

            if opt_get(self.opt, ["network_G", "flow", "stackRRDB", "concat"]) or False:
                keys = ["last_lr_fea", "fea_up1", "fea_up2"]
                if "fea_up0" in rrdbResults.keys():
                    keys.append("fea_up0")
                if "fea_up-1" in rrdbResults.keys():
                    keys.append("fea_up-1")
                if "fea_up-2" in rrdbResults.keys():
                    keys.append("fea_up-2")
                if self.opt["scale"] >= 4:
                    keys.append("fea_up4")
                if self.opt["scale"] >= 8:
                    keys.append("fea_up8")
                if self.opt["scale"] == 16:
                    keys.append("fea_up16")
                for k in keys:
                    h = rrdbResults[k].shape[2]
                    w = rrdbResults[k].shape[3]
                    rrdbResults[k] = torch.cat(
                        [rrdbResults[k], F.interpolate(concat, (h, w))], dim=1
                    )
        return rrdbResults

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(
            z ** 2, dim=[1, 2, 3]
        ) - z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(
        self,
        lr,
        z,
        y_onehot,
        eps_std,
        epses=None,
        lr_enc=None,
        add_gt_noise=True,
        std=None,
    ):
        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.opt["scale"] ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None:
            lr_enc = self.rrdbPreprocessing(lr)

        x, logdet = self.flowUpsamplerNet(
            rrdbResults=lr_enc,
            z=z,
            eps_std=eps_std,
            reverse=True,
            epses=epses,
            logdet=logdet,
            std=std,
        )

        return x, logdet