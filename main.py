
import numpy as np, argparse, time, random


from model import *

from trainer import train_or_eval_model

from dataloader import get_data_loaders
from transformers import AdamW



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



if __name__ == '__main__':

    path = './saved_models/'  # 日志 模型保存路径

    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')

    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')
    parser.add_argument('--dataset_name', default='MELD', type=str, help='dataset name, IEMOCAP, MELD, DailyDialog, EmoryNLP')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate')  #####
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help='batch size') ##


    parser.add_argument('--seed', type=int, default=100, help='random seed') ##


    args = parser.parse_args()

    print(args)

    # 固定随机种子
    seed_everything(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)

    device = args.device
    n_epochs = args.epochs
    batch_size = args.batch_size


    train_loader, valid_loader, test_loader = get_data_loaders(
        dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)


    if 'IEMOCAP' in args.dataset_name:
        n_classes = 6
    else:
        n_classes = 7

    print('building model..')
    
    model = DualGATs(args, n_classes)
 
    if torch.cuda.device_count() > 1:
        print('Multi-GPU...........')
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model.to(device)

    
    loss_function = nn.CrossEntropyLoss(ignore_index=-1) # 忽略掉label=-1 的类
    

    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_fscore, best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_acc = 0.
    best_fscore = 0.

    best_model = None
    for e in range(n_epochs):  # 遍历每个epoch
        start_time = time.time()

        train_loss, train_acc, _, _, train_fscore = train_or_eval_model(model, loss_function,
                                                                        train_loader, device,
                                                                        args, optimizer, True)
        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_model(model, loss_function,
                                                                        valid_loader, device, args)
        test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_model(model, loss_function,
                                                                                      test_loader, device, args)

        all_fscore.append([valid_fscore, test_fscore])

        print(
            'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
                   test_acc,
                   test_fscore, round(time.time() - start_time, 2)))

        e += 1


    print('finish training!')


    all_fscore = sorted(all_fscore, key=lambda x: (x[0], x[1]), reverse=True)  # 优先按照验证集 f1 进行排序

    print('Best val F-Score:{}'.format(all_fscore[0][0]))  # 验证集最好性能 
    print('Best test F-Score based on validation:{}'.format(all_fscore[0][1]))  # 验证集取得最好性能时 对应测试集的下性能
    print('Best test F-Score based on test:{}'.format(max([f[1] for f in all_fscore])))  # 测试集 最好的性能

