from distutils.command.config import config
import random
import torch
import numpy as np
import torch.optim
from time import time
import os
from utility.load_data import *
from sklearn.metrics import roc_auc_score
from Model.Lightgcn import LightGCN
from utility.compute import *


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.regs = 0
        self.embed_size = 32
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'BookCrossing'
        self.attack = '0.01'
        self.layer_size = '[64,64]'
        self.verbose = 1
        self.Ks = '[10]'
        self.data_type = 'full'
        self.init_std = 1e-4
        self.seed = 1024

        # lightgcn hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 10
        self.A_split = False
        self.dropout = False
        self.pretrain = 0

    def reset(self, config):
        for name, val in config.items():
            setattr(self, name, val)


class early_stoper(object):
    def __init__(self, refer_metric='valid_auc', stop_condition=10):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric

    def update_and_isbest(self, eval_metric, epoch):
        if self.init_flag:
            self.best_epoch = epoch
            self.init_flag = False
            self.best_eval_result = eval_metric
            return True
        else:
            if eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]:
                self.best_eval_result = eval_metric
                self.not_change = 0
                self.best_epoch = epoch
                return True
            else:
                self.not_change += 1
                return False

    def is_stop(self):
        if self.not_change > self.stop_condition:
            return True
        else:
            return False


def main(config_args):
    args = model_hyparameters()
    assert config_args is not None
    args.reset(config_args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_generator = Data_for_LightGCN(args, path=args.data_path + args.dataset + '/' + args.attack)
    data_generator.set_train_mode(mode=args.data_type)

    save_name = 'LightGCN_'
    for name_str, name_val in config_args.items():
        save_name += name_str + '-' + str(name_val) + '-'

    model = LightGCN(args, dataset=data_generator).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dataloader = data_generator.generate_train_dataloader(batch_size=args.batch_size)

    e_stoper = early_stoper(refer_metric='valid_auc', stop_condition=10)
    mask = get_eval_mask(data_generator)
    total_time = 0


    top_k_list = [10, 20]

    for epoch in range(args.epoch):
        t1 = time()
        bce_loss = 0
        for batch_i, batch_data in enumerate(train_dataloader):
            opt.zero_grad()
            batch_user, batch_item, batch_label = batch_data[:, 0].cuda().long(), batch_data[:, 1].cuda().long(), batch_data[:, -1].cuda().float()
            bce_loss = model.compute_bce_loss(batch_user, batch_item, batch_label)
            bce_loss.backward()
            opt.step()
            bce_loss += bce_loss.item()

        bce_loss /= batch_i

        t2 = time()

        valid_auc, valid_auc_or, valid_auc_and, \
        test_auc, test_auc_or, test_auc_and, \
        valid_topn, test_topn = get_eval_result(data_generator, model, mask, top_k_list)


        valid_str = format_topn_metrics(valid_topn, top_k_list)
        test_str = format_topn_metrics(test_topn, top_k_list)

        t3 = time()

        print(f"[Epoch {epoch}] time:{t3-t2:.4f}")
        print(f"Valid AUC: {valid_auc:.4f} | OR: {valid_auc_or:.4f} | AND: {valid_auc_and:.4f}")
        print(f"Valid TopN | {valid_str}")
        print(f"Test  AUC: {test_auc:.4f} | OR: {test_auc_or:.4f} | AND: {test_auc_and:.4f}")
        print(f"Test  TopN | {test_str}")
        print("-" * 80)

        one_result = {'valid_auc': valid_auc, 'test_auc': test_auc}
        is_best = e_stoper.update_and_isbest(one_result, epoch)

        if is_best:
            total_time += (t2 - t1)
            print('save best model')
            torch.save(model.state_dict(), './Weights/LightGCN/' + save_name + ".pth")

        if e_stoper.is_stop():
            print("early stop condiction reached at epoch:", epoch)
            break

    print(f"total: {total_time:.6f} seconds")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = {
        'lr': 1e-4, # [1e-2, 1e-3, 1e-4]
        'embed_size': 32, # [32, 48, 64]
        'batch_size': 2048,
        'data_type': 'retraining',
        'dataset': 'Amazon',  # [Amazon, BookCrossing, Mooccube, mooper]
        'attack': '0.01',
        'seed': 1024,
        'init_std': 1e-4,  # [1e-2, 1e-3, 1e-4]
    }
    main(config)