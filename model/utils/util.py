from __future__ import print_function
import torch
from PIL import Image
import numpy as np
import os
import yaml
import torch.nn.init as init
import math
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch import autograd
import os.path as osp

def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE):
    # print "real_data: ", real_data.size(), fake_data.size()
    LAMBDA = 10
    grad_outputs = []
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 256, 128)
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    for i, tensor in enumerate(disc_interpolates):
        ones = torch.ones(tensor.size()).cuda()
        grad_outputs.append(ones)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_checkpoint(iterations, model, optimizer, loss, path):
    # save checkpoint
    save_path = path
    torch.save({
        'iterations': iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def save_img(imgs, save_path, batch):
    torchvision.utils.save_image(imgs, save_path, nrow=batch)


def denormalize_recon(x):
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    x_recon = (x * std) + mean

    return x_recon


def UnNormalize(tensor):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])

    inv_tensor = invTrans(tensor)
    return inv_tensor


def assign_adain_params(dim, adain_params_w, adain_params_b, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params_b[:, :dim].contiguous()
            std = adain_params_w[:, :dim].contiguous()
            m.bias = mean.view(-1)
            m.weight = std.view(-1)
            if adain_params_w.size(1) > dim:  # Pop the parameters
                adain_params_b = adain_params_b[:, dim:]
                adain_params_w = adain_params_w[:, dim:]

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2 * m.num_features
    return num_adain_params


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))