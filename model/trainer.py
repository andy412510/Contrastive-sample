"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn
import random
from torch.nn import functional as F
from model.networks import Generator, MsImageDis, MLP, get_scheduler, Encoder_pose
from model.losses import GANLoss_MUNIT
from model.utils.util import assign_adain_params, save_checkpoint, save_img, UnNormalize, calc_gradient_penalty, \
                            load_checkpoint, weights_init, mkdir
from model import models
from model.models.memory import Memory
from model.evaluators import Evaluator, extract_features
from collections import OrderedDict


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.config = args
        self.config_gen = args['gen']
        self.config_dis = args['dis']

        self._init_models()
        self._init_losses()
        self._init_cross_optimizers()
        self.mAP = 0
        mkdir(self.config['checkpoint_path'])
        mkdir(self.config['output_path'])

    def _init_models(self):
        self.memory = Memory(num_features=2048, num_samples=self.config['num_samples'], temp=self.config['temperature'],
                             momentum=self.config['momentum'], K=self.config['K'])
        # Di For source
        self.net_Di = MsImageDis(self.config['input_dim'], self.config_dis)
        # Encoder
        self.net_E_c = models.create('ft_net', num_features=0, dropout=0, num_classes=0)
        self.net_E_pose = Encoder_pose(self.config_gen['n_downsample'], self.config_gen['n_res'],
                                       self.config['pose_dim'],
                                       self.config_gen['dim'], 'in', self.config_gen['activ'],
                                       pad_type=self.config_gen['pad_type'],
                                       dropout=self.config_gen['dropout'], tanh=self.config_gen['tanh'],
                                       res_type='basic')
        self.output_dim = self.net_E_pose.output_dim
        # G For source
        self.net_G = Generator(self.config_gen['n_downsample'], self.config_gen['n_res'], self.output_dim,
                               3, dropout=self.config_gen['dropout'], res_norm='adain', activ=self.config_gen['activ'],
                               pad_type=self.config_gen['pad_type'], non_local=self.config_gen['non_local'], fp16=False)
        # MLP to generate G AdaIN parameters
        self.mlp_w1 = MLP(self.config_gen['id_dim'], 2 * self.output_dim, self.config_gen['mlp_dim'], 3,
                          norm=self.config_gen['mlp_norm'], activ=self.config_gen['activ'])
        self.mlp_w2 = MLP(self.config_gen['id_dim'], 2 * self.output_dim, self.config_gen['mlp_dim'], 3,
                          norm=self.config_gen['mlp_norm'], activ=self.config_gen['activ'])
        self.mlp_w3 = MLP(self.config_gen['id_dim'], 2 * self.output_dim, self.config_gen['mlp_dim'], 3,
                          norm=self.config_gen['mlp_norm'], activ=self.config_gen['activ'])
        self.mlp_w4 = MLP(self.config_gen['id_dim'], 2 * self.output_dim, self.config_gen['mlp_dim'], 3,
                          norm=self.config_gen['mlp_norm'], activ=self.config_gen['activ'])

        self.mlp_b1 = MLP(self.config_gen['id_dim'], 2 * self.output_dim, self.config_gen['mlp_dim'], 3,
                          norm=self.config_gen['mlp_norm'], activ=self.config_gen['activ'])
        self.mlp_b2 = MLP(self.config_gen['id_dim'], 2 * self.output_dim, self.config_gen['mlp_dim'], 3,
                          norm=self.config_gen['mlp_norm'], activ=self.config_gen['activ'])
        self.mlp_b3 = MLP(self.config_gen['id_dim'], 2 * self.output_dim, self.config_gen['mlp_dim'], 3,
                          norm=self.config_gen['mlp_norm'], activ=self.config_gen['activ'])
        self.mlp_b4 = MLP(self.config_gen['id_dim'], 2 * self.output_dim, self.config_gen['mlp_dim'], 3,
                          norm=self.config_gen['mlp_norm'], activ=self.config_gen['activ'])

        self.net_G.cuda()
        self.net_E_c.cuda()
        self.net_E_pose.cuda()
        self.mlp_w1.cuda()
        self.mlp_w2.cuda()
        self.mlp_w3.cuda()
        self.mlp_w4.cuda()
        self.mlp_b1.cuda()
        self.mlp_b2.cuda()
        self.mlp_b3.cuda()
        self.mlp_b4.cuda()
        self.net_Di.cuda()
        self.evaluator = Evaluator(self.net_E_c)

    def _init_losses(self):
        if self.config_dis['smooth_label']:
            self.rand_list = [True] * 1 + [False] * 10000
        else:
            self.rand_list = [False]
        self.criterionGAN_D = GANLoss_MUNIT(smooth=self.config_dis['smooth_label']).cuda()
        self.criterionGAN_G = GANLoss_MUNIT(smooth=False).cuda()
        self.criterion_moco = nn.CrossEntropyLoss().cuda()

    def _init_cross_optimizers(self):
        param_groups_g = [
            {'params': self.net_G.parameters()},
            {'params': self.net_E_pose.parameters()},
            {'params': self.mlp_w1.parameters()},
            {'params': self.mlp_w2.parameters()},
            {'params': self.mlp_w3.parameters()},
            {'params': self.mlp_w4.parameters()},
            {'params': self.mlp_b1.parameters()},
            {'params': self.mlp_b2.parameters()},
            {'params': self.mlp_b3.parameters()},
            {'params': self.mlp_b4.parameters()},
        ]
        param_groups_d = [
            {'params': self.net_Di.parameters()}
        ]
        param_groups_E_c = [{'params': self.net_E_c.parameters()}]

        self.optimizer_G = torch.optim.Adam(param_groups_g,
                                             lr=self.config['lr_g'], betas=(self.config['beta1'], self.config['beta2']),
                                             weight_decay=self.config['weight_decay'])
        self.optimizer_D = torch.optim.Adam(param_groups_d,
                                             lr=self.config['lr_d'], betas=(self.config['beta1'], self.config['beta2']),
                                             weight_decay=self.config['weight_decay'])
        self.optimizer_E_c = torch.optim.SGD(param_groups_E_c, lr=self.config['lr_id'],
                                             weight_decay=self.config['weight_decay'], momentum=0.9, nesterov=True)

        self.dis_scheduler = get_scheduler(self.optimizer_D, self.config)
        self.gen_scheduler = get_scheduler(self.optimizer_G, self.config)
        self.id_scheduler = get_scheduler(self.optimizer_E_c, self.config)
        self.id_scheduler.gamma = self.config['gamma']

    def _init_models_status(self, cluster_loader):
        E_c_checkpoint = load_checkpoint(self.config['E_c_init'])
        new_state_dict = OrderedDict()
        for key, value in E_c_checkpoint.items():  # 把 netE_pretrain keys 前面都+ model.
            name = 'module.' + key
            new_state_dict[name] = value  # netE_pretrain keys 前面都+ model後叫做 new_state_dict
        self.net_E_c.load_state_dict(E_c_checkpoint, strict=False)
        print('Finish E_c init.')
        if self.config['stage'] == 2:
            self.net_E_c.eval()
            self.net_G.apply(weights_init(self.config_gen['init']))
            self.mlp_w1.apply(weights_init(self.config_gen['init']))
            self.mlp_w2.apply(weights_init(self.config_gen['init']))
            self.mlp_w3.apply(weights_init(self.config_gen['init']))
            self.mlp_w4.apply(weights_init(self.config_gen['init']))
            self.mlp_b1.apply(weights_init(self.config_gen['init']))
            self.mlp_b2.apply(weights_init(self.config_gen['init']))
            self.mlp_b3.apply(weights_init(self.config_gen['init']))
            self.mlp_b4.apply(weights_init(self.config_gen['init']))

        if self.config['stage'] == 3:
            # load checkpoints
            G_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'net_G_best.pth')['model_state_dict']
            self.net_G.load_state_dict(G_checkpoint, strict=True)  # strict=True: totally same
            Ep_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'net_E_pose_best.pth')['model_state_dict']
            self.net_E_pose.load_state_dict(Ep_checkpoint, strict=True)
            D_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'net_Di_best.pth')['model_state_dict']
            self.net_Di.load_state_dict(D_checkpoint, strict=True)
            b1_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'mlp_b1_best.pth')['model_state_dict']
            self.mlp_b1.load_state_dict(b1_checkpoint, strict=True)
            b2_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'mlp_b2_best.pth')['model_state_dict']
            self.mlp_b2.load_state_dict(b2_checkpoint, strict=True)
            b3_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'mlp_b3_best.pth')['model_state_dict']
            self.mlp_b3.load_state_dict(b3_checkpoint, strict=True)
            b4_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'mlp_b4_best.pth')['model_state_dict']
            self.mlp_b4.load_state_dict(b4_checkpoint, strict=True)
            w1_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'mlp_w1_best.pth')['model_state_dict']
            self.mlp_w1.load_state_dict(w1_checkpoint, strict=True)
            w2_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'mlp_w2_best.pth')['model_state_dict']
            self.mlp_w2.load_state_dict(w2_checkpoint, strict=True)
            w3_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'mlp_w3_best.pth')['model_state_dict']
            self.mlp_w3.load_state_dict(w3_checkpoint, strict=True)
            w4_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'mlp_w4_best.pth')['model_state_dict']
            self.mlp_w4.load_state_dict(w4_checkpoint, strict=True)
            self._init_memory(cluster_loader)

        self.net_G.train()
        self.net_E_pose.train()
        self.mlp_w1.train()
        self.mlp_w2.train()
        self.mlp_w3.train()
        self.mlp_w4.train()
        self.mlp_b1.train()
        self.mlp_b2.train()
        self.mlp_b3.train()
        self.mlp_b4.train()
        self.net_Di.train()

    def _init_memory(self, cluster_loader):
        dict_f1, label = extract_features(self.net_E_c, cluster_loader, print_freq=50)
        cf = torch.stack(list(dict_f1.values()))
        labels = torch.stack(list(label.values()))
        self.memory.features = cf.cuda()
        self.memory.labels = labels.cuda()
        print('Memory initialized.')

    def encode(self, input, pose):
        # encode an image to its content and pose feature
        content, f = self.net_E_c(input, mode='fix')
        pose_feature = self.net_E_pose(pose)
        return content, pose_feature, f

    def decode(self, content, pose, G, w1,w2,w3,w4,b1,b2,b3,b4):
        # decode style codes to an image
        ID1 = content[:, :2048]
        ID2 = content[:, 2048:4096]
        ID3 = content[:, 4096:6144]
        ID4 = content[:, 6144:]
        adain_params_w = torch.cat((w1(ID1), w2(ID2), w3(ID3), w4(ID4)), 1)
        adain_params_b = torch.cat((b1(ID1), b2(ID2), b3(ID3), b4(ID4)), 1)
        assign_adain_params(self.output_dim, adain_params_w, adain_params_b, G)
        images = G(pose)
        return images

    def set_inputs(self, data, iterations, epoch):
        self.iterations = iterations
        self.epoch = epoch
        self.input = data['img'].cuda()
        self.pose = data['mesh_org'].cuda()
        self.index = data['index'].cuda()
        self.pose_nv = data['mesh_nv'].cuda()
        self.pose_all = data['mesh_all']
        self.train_index = data['index'].cuda()

    def forward(self):
        self.content, pose_feature, self.f = self.encode(self.input, pose=self.pose)
        max = torch.tensor(100)
        # specific
        for i, pose in enumerate(self.pose_all):
            pose = pose.cuda()
            _, posesp_feature, _ = self.encode(self.input, pose=pose)
            # distance in feature domain or image domain?
            s2t = self.decode(self.content, posesp_feature, self.net_G, self.mlp_w1, self.mlp_w2, self.mlp_w3,
                              self.mlp_w4, self.mlp_b1, self.mlp_b2, self.mlp_b3, self.mlp_b4)
            distance = F.l1_loss(self.input, s2t).detach()
            if distance < max:
                max = distance
                self.x_s2t = s2t

        # random
        # _, posenv_feature, _ = self.encode(self.input, pose=self.pose_nv)
        # self.x_s2t = self.decode(self.content, posenv_feature, self.net_G, self.mlp_w1, self.mlp_w2, self.mlp_w3, self.mlp_w4,
        #                     self.mlp_b1, self.mlp_b2, self.mlp_b3, self.mlp_b4)

        self.content_s2t, _, self.f_s2t = self.encode(self.x_s2t, pose=self.pose)
        self.x_s2s = self.decode(self.content, pose_feature, self.net_G, self.mlp_w1, self.mlp_w2, self.mlp_w3,
                            self.mlp_w4, self.mlp_b1, self.mlp_b2, self.mlp_b3, self.mlp_b4)
        self.content_s2s, _, self.f_s2s = self.encode(self.x_s2s, pose=self.pose)
        self.x_s2t2s = self.decode(self.content_s2t, pose_feature, self.net_G, self.mlp_w1, self.mlp_w2, self.mlp_w3,
                                 self.mlp_w4, self.mlp_b1, self.mlp_b2, self.mlp_b3, self.mlp_b4)
        self.content_s2t2s, _, _ = self.encode(self.x_s2t2s, pose=self.pose)


        self.net_E_c.train()

    def backward_D(self):
        pred_real = self.net_Di(self.input)
        pred_fake = self.net_Di(self.x_s2t.detach())
        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True) * self.config['gan_w']
            loss_D_fake = self.criterionGAN_D(pred_real, False) * self.config['gan_w']
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True) * self.config['gan_w']
            loss_D_fake = self.criterionGAN_D(pred_fake, False) * self.config['gan_w']
        loss_D = (loss_D_real + loss_D_fake) * 0.5  # wgan-gp D loss weight = 10
        loss_D.backward()
        gradient_penalty = calc_gradient_penalty(self.net_Di, self.input, self.x_s2t, self.config['batch_size'])
        gradient_penalty.backward()
        self.loss_Di = loss_D.item()

    def backward_G(self):
        # # GAN loss
        pred = self.net_Di(self.x_s2t)
        loss_gen_adv = self.criterionGAN_G(pred, True)
        # reconstruction loss
        loss_recon_s2s = F.l1_loss(self.x_s2s, self.input) if self.config['recon_x_w'] > 0 else 0
        loss_recon_s2t2s = F.l1_loss(self.x_s2t2s, self.input) if self.config['recon_x_w'] > 0 else 0
        # content specific loss
        loss_content_s2t = F.l1_loss(self.content_s2t, self.content) if self.config['recon_c_w'] > 0 else 0
        loss_content_s2s = F.l1_loss(self.content_s2s, self.content) if self.config['recon_c_w'] > 0 else 0
        loss_content_s2t2s = F.l1_loss(self.content_s2t2s, self.content) if self.config['recon_c_w'] > 0 else 0
        if self.config['stage'] == 2:
            self.loss_memory = loss_memory = 0
        elif self.config['stage'] == 3:
            loss_memory = self.memory(F.normalize(self.f), F.normalize(self.f_s2t), self.train_index) * self.config['contrastive_w']
            self.loss_memory = loss_memory.item()
        loss_G = self.config['gan_w'] * loss_gen_adv + \
                         self.config['recon_x_w'] * loss_recon_s2s + \
                         self.config['recon_x_cyc_w'] * loss_recon_s2t2s + \
                         self.config['recon_c_w'] * loss_content_s2t + \
                         self.config['recon_c_w'] * loss_content_s2t2s + \
                         self.config['recon_c_w'] * loss_content_s2s

        loss_gen = loss_G + loss_memory
        loss_gen.backward()
        # total loss
        self.loss_G = loss_G.item()
        self.loss_gen_adv = loss_gen_adv.item()
        self.loss_recon_s2s = loss_recon_s2s.item()
        self.loss_recon_s2t2s = loss_recon_s2t2s.item()
        self.loss_content_s2t = loss_content_s2t.item()
        self.loss_content_s2s = loss_content_s2s.item()
        self.loss_content_s2t2s = loss_content_s2t2s.item()

    def visdom_plot(self, viz, mAP):
        iterations = self.iterations
        with torch.no_grad():
            self.total_loss = self.loss_G + self.loss_Di + self.loss_memory
            if iterations == 0:
                viz.line([0.], [0], win='mAP', opts=dict(title='mAP', xlabel='Iterations'))
                viz.line([0.], [0], win='total_loss', opts=dict(title='total_loss', xlabel='Iterations'))
                viz.line([0.], [0], win='loss_G', opts=dict(title='loss_G', xlabel='Iterations'))
                viz.line([0.], [0], win='loss_memory', opts=dict(title='loss_memory', xlabel='Iterations'))
                # adv
                viz.line([0.], [0], win='loss_Di', opts=dict(title='loss_Di', xlabel='Iterations'))
                viz.line([0.], [0], win='loss_gen_adv', opts=dict(title='loss_gen_adv_Dt', xlabel='Iterations'))
                # recon
                viz.line([0.], [0], win='loss_recon_s2s', opts=dict(title='loss_recon_s2s', xlabel='Iterations'))
                viz.line([0.], [0], win='loss_recon_s2t2s', opts=dict(title='loss_recon_s2t2s', xlabel='Iterations'))
                # content
                viz.line([0.], [0], win='loss_content_s2t',
                         opts=dict(title='loss_content_s2t', xlabel='Iterations'))
                viz.line([0.], [0], win='loss_content_s2s',
                         opts=dict(title='loss_content_s2s', xlabel='Iterations'))
                viz.line([0.], [0], win='loss_content_s2t2s',
                         opts=dict(title='loss_content_s2t2s', xlabel='Iterations'))
            # viz.line(Y軸的下一個點, X軸的下一個點, 視窗名稱(視窗右上角顯示的), 加到下一個點之後)
            if iterations % self.config['loss_freq'] == 0:
                viz.line([mAP], [iterations], win='mAP', update='append')
                viz.line([self.total_loss], [iterations], win='total_loss', update='append')
                viz.line([self.loss_G], [iterations], win='loss_G', update='append')
                viz.line([self.loss_memory], [iterations], win='loss_memory', update='append')
                # adv
                viz.line([self.loss_Di], [iterations], win='loss_Di', update='append')
                viz.line([self.loss_gen_adv], [iterations], win='loss_gen_adv', update='append')
                # recon
                viz.line([self.loss_recon_s2s], [iterations], win='loss_recon_s2s', update='append')
                viz.line([self.loss_recon_s2t2s], [iterations], win='loss_recon_s2t2s', update='append')
                # content
                viz.line([self.loss_content_s2t], [iterations], win='loss_content_s2t', update='append')
                viz.line([self.loss_content_s2s], [iterations], win='loss_content_s2s', update='append')
                viz.line([self.loss_content_s2t2s], [iterations], win='loss_content_s2t2s', update='append')
        return self.total_loss

    def checkpoint_G(self, iterations, best_loss, batch):
        if self.total_loss < best_loss:
            # D
            save_checkpoint(iterations, self.net_Di, self.optimizer_D, self.total_loss,
                            self.config['checkpoint_path'] + 'net_Di_best.pth')
            # E
            save_checkpoint(iterations, self.net_E_pose, self.optimizer_E_c, self.total_loss,
                            self.config['checkpoint_path'] + 'net_E_pose_best.pth')
            save_checkpoint(iterations, self.net_E_c, self.optimizer_E_c, self.total_loss,
                            self.config['checkpoint_path'] + 'net_E_c_best.pth')
            # G
            save_checkpoint(iterations, self.net_G, self.optimizer_G, self.total_loss,
                            self.config['checkpoint_path'] + 'net_G_best.pth')
            # MLP
            save_checkpoint(iterations, self.mlp_w1, self.optimizer_G, self.total_loss,
                            self.config['checkpoint_path'] + 'mlp_w1_best.pth')
            save_checkpoint(iterations, self.mlp_w2, self.optimizer_G, self.total_loss,
                            self.config['checkpoint_path'] + 'mlp_w2_best.pth')
            save_checkpoint(iterations, self.mlp_w3, self.optimizer_G, self.total_loss,
                            self.config['checkpoint_path'] + 'mlp_w3_best.pth')
            save_checkpoint(iterations, self.mlp_w4, self.optimizer_G, self.total_loss,
                            self.config['checkpoint_path'] + 'mlp_w4_best.pth')
            save_checkpoint(iterations, self.mlp_b1, self.optimizer_G, self.total_loss,
                            self.config['checkpoint_path'] + 'mlp_b1_best.pth')
            save_checkpoint(iterations, self.mlp_b2, self.optimizer_G, self.total_loss,
                            self.config['checkpoint_path'] + 'mlp_b2_best.pth')
            save_checkpoint(iterations, self.mlp_b3, self.optimizer_G, self.total_loss,
                            self.config['checkpoint_path'] + 'mlp_b3_best.pth')
            save_checkpoint(iterations, self.mlp_b4, self.optimizer_G, self.total_loss,
                            self.config['checkpoint_path'] + 'mlp_b4_best.pth')
            best_loss = self.total_loss
            print('best loss', best_loss)
            # save imgs
            with torch.no_grad():
                source_imgs = torch.cat([UnNormalize(self.input[:4]), UnNormalize(self.x_s2t[:4])], dim=0)
                save_img(source_imgs, self.config['output_path'] + 'source_best.png', batch)
                recon_imgs = torch.cat([UnNormalize(self.x_s2s[:4]), UnNormalize(self.x_s2t2s[:4])], dim=0)
                save_img(recon_imgs, self.config['output_path'] + 'recon_best.png', batch)
        return best_loss

    def checkpoint(self, batch, test_loader, dataset_target):
        iterations = self.epoch
        _, mAP = self.evaluator.evaluate(test_loader, dataset_target.query, dataset_target.gallery, cmc_flag=True)
        if mAP > self.mAP:
            print('best mAP in epoch.', iterations)
            self.mAP = mAP
            # D
            save_checkpoint(iterations, self.net_Di, self.optimizer_D, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_net_Di_best.pth')
            # E
            save_checkpoint(iterations, self.net_E_c, self.optimizer_E_c, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_net_E_c_best.pth')
            save_checkpoint(iterations, self.net_E_pose, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_net_E_pose_best.pth')
            # G
            save_checkpoint(iterations, self.net_G, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_net_G_best.pth')
            # MLP
            save_checkpoint(iterations, self.mlp_w1, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_mlp_w1_best.pth')
            save_checkpoint(iterations, self.mlp_w2, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_mlp_w2_best.pth')
            save_checkpoint(iterations, self.mlp_w3, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_mlp_w3_best.pth')
            save_checkpoint(iterations, self.mlp_w4, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_mlp_w4_best.pth')
            save_checkpoint(iterations, self.mlp_b1, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_mlp_b1_best.pth')
            save_checkpoint(iterations, self.mlp_b2, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_mlp_b2_best.pth')
            save_checkpoint(iterations, self.mlp_b3, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_mlp_b3_best.pth')
            save_checkpoint(iterations, self.mlp_b4, self.optimizer_G, self.mAP,
                            self.config['checkpoint_path'] + 'stage3_mlp_b4_best.pth')
            # save imgs
            with torch.no_grad():
                source_imgs = torch.cat([UnNormalize(self.input[:4]), UnNormalize(self.x_s2t[:4])], dim=0)
                save_img(source_imgs, self.config['output_path'] + 'source_best.png', batch)
        return self.mAP, mAP

    def resume(self):
        # load checkpoints
        G_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_net_G_best.pth')['model_state_dict']
        self.net_G.load_state_dict(G_checkpoint, strict=True)  # strict=True: totally same
        Ec_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_net_E_c_best.pth')[
            'model_state_dict']
        self.net_E_c.load_state_dict(Ec_checkpoint, strict=True)
        Ep_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_net_E_pose_best.pth')[
            'model_state_dict']
        self.net_E_pose.load_state_dict(Ep_checkpoint, strict=True)
        D_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_net_Di_best.pth')['model_state_dict']
        self.net_Di.load_state_dict(D_checkpoint, strict=True)
        b1_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_mlp_b1_best.pth')['model_state_dict']
        self.mlp_b1.load_state_dict(b1_checkpoint, strict=True)
        b2_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_mlp_b2_best.pth')['model_state_dict']
        self.mlp_b2.load_state_dict(b2_checkpoint, strict=True)
        b3_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_mlp_b3_best.pth')['model_state_dict']
        self.mlp_b3.load_state_dict(b3_checkpoint, strict=True)
        b4_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_mlp_b4_best.pth')['model_state_dict']
        self.mlp_b4.load_state_dict(b4_checkpoint, strict=True)
        w1_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_mlp_w1_best.pth')['model_state_dict']
        self.mlp_w1.load_state_dict(w1_checkpoint, strict=True)
        w2_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_mlp_w2_best.pth')['model_state_dict']
        self.mlp_w2.load_state_dict(w2_checkpoint, strict=True)
        w3_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_mlp_w3_best.pth')['model_state_dict']
        self.mlp_w3.load_state_dict(w3_checkpoint, strict=True)
        w4_checkpoint = load_checkpoint(self.config['checkpoint_path'] + 'stage3_mlp_w4_best.pth')['model_state_dict']
        self.mlp_w4.load_state_dict(w4_checkpoint, strict=True)

        # Load optimizers
        # Reinitilize schedulers
        # self.dis_scheduler = get_scheduler(self.dis_opt, config, iterations)
        # self.gen_scheduler = get_scheduler(self.gen_opt, config, iterations)

    def sample_recon(self, x_img, pose_all):
        self.net_G.eval()
        self.net_E_c.eval()
        self.net_E_pose.eval()
        self.mlp_b1.eval()
        self.mlp_b2.eval()
        self.mlp_b3.eval()
        self.mlp_b4.eval()
        self.mlp_w1.eval()
        self.mlp_w2.eval()
        self.mlp_w3.eval()
        self.mlp_w4.eval()
        with torch.no_grad():
            max = torch.tensor(100)
            # specific
            for i, pose in enumerate(pose_all):
                pose = pose.cuda()
                content, posesp_feature, _ = self.encode(x_img, pose=pose)
                # distance in feature domain or image domain?
                s2t = self.decode(content, posesp_feature, self.net_G, self.mlp_w1, self.mlp_w2, self.mlp_w3,
                                  self.mlp_w4, self.mlp_b1, self.mlp_b2, self.mlp_b3, self.mlp_b4)
                distance = F.l1_loss(x_img, s2t).detach()
                if distance < max:
                    max = distance
                    x_recon = s2t

        return x_recon

    def evaluate(self, test_loader, dataset_target):
        _, mAP_1 = self.evaluator.evaluate(test_loader, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    def optimize_cross_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.optimizer_E_c.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        if self.config['stage'] == 3:
            self.optimizer_E_c.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.id_scheduler is not None:
            self.id_scheduler.step()
