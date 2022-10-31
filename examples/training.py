from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
from visdom import Visdom
from tqdm import tqdm
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from model import datasets
from model.utils.data import transforms as T
from model.utils.util import get_config
from model.trainer import Trainer
from model.utils.data.preprocessor import Preprocessor

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def main(args):
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = get_config(args.config)
    iterations = 0
    best_loss = 100
    mAP = 0
    best_mAP = 0
    viz = Visdom(port=args.port)  # 將visdom實體化  python -m visdom.server -port=8098

    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, config['height'], config['width'], config['batch_size'] * 8,
                                  args.workers)

    # Create your data loader here.
    # We need three data loader in this project: train_loader, cluster_loader and test_loader.

    # train_loader:
    # The training data of dataset. The label information should be produced by unsupervised way.
    # We follow [1] to use DBSCAN to produce pseudo labels in each epoch.
    # The return data of train_loader should be:
    # return {'img': img,  # the images of training data [batch, 3, 256, 128]
    #         'mesh_org': mesh_org,  # the original pose image of input data [batch, 1, 256, 128]
    #         'mesh_nv': mesh_nv,  # a random view pose image of input data through 7 views [batch, 1, 256, 128]
    #         'mesh_all': mesh_all,  # 7 view images of input data [list: 7]
    #         'fname': fname,  # input file paths [list: batch]
    #         'pid': pid,  # input pseudo labels [Tensor: batch]
    #         'index': index  # index in dataset [Tensor: batch]
    #         }

    # cluster_loader:
    # The training data without pose information. These data is used for memory bank.
    # The return data of train_loader should be:
    # return img, fname, pid, camid, index
    # img: the images of training data [batch*8, 3, 256, 128]
    # fname: input file paths [list: batch*8]
    # pid: input pseudo labels [Tensor: batch*8]
    # camid: not use
    # index: index in dataset [Tensor: batch*8]

    # [1] Kuan Zhu, Haiyun Guo, Tianyi Yan, Yousong Zhu, Jin-qiao Wang, and Ming Tang,
    #     “Pass: Part-aware self-supervised pre-training for person re-identification,”
    #     in European Conference on Computer Vision (ECCV). Springer, 2022, pp. 198–214.

    model = Trainer(config)
    model._init_models_status(cluster_loader)
    model.evaluate(test_loader, dataset)  # check if Ec load checkpoint correct
    for epoch in range(config['start_epoch'], config['epoch']):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        pbar.set_description(f'Eopch {epoch}/{config["epoch"]}')
        for it, data in enumerate(train_loader):  # (batch, 3, 256, 128)
            model.set_inputs(data=data, iterations=iterations, epoch=epoch)
            model.optimize_cross_parameters()
            loss = model.visdom_plot(viz, mAP)
            if config['stage'] == 2:
                best_loss = model.checkpoint_G(iterations, best_loss, config['batch_size'])
            pbar.update(1)
            pbar.set_postfix(loss=loss, best_mAP=best_mAP)
            iterations += 1
        if config['stage'] == 3:
            best_mAP, mAP = model.checkpoint(config['batch_size'], test_loader, dataset)
        model.update_learning_rate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")
    # gan config
    parser.add_argument('--config', type=str, default='../configs/config.yaml',
                        help='Path to the config file.')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmc-reid',  # market1501, dukemtmc-reid
                        choices=datasets.names())
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--port', type=int, default=8069)
    parser.add_argument("--resume", action="store_true")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/andy/examples/data/')
    parser.add_argument('--mesh-dir', type=str, metavar='PATH',
                        default='/home/andy/examples/mesh/DukeMTMC/')  # DukeMTMC, market
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
