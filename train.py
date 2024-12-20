import numpy as np
import os
import torch
from torch import nn
import argparse
from utils import setup_seed, load_har_data, count_FLOPS
from model import CNN, MobileNet, RepMobile, reparameterize_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train a HAR task')
    parser.add_argument(
        '--dataset',
        help='select dataset',
        default='uci'
        )
    parser.add_argument(
        '--model',
        help='select network',
        default='cnn'
        )
    parser.add_argument('--batch', type=int, help='batch_size', default=128)
    parser.add_argument('--epoch', type=int, help='epoch', default=100)
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.0005)
    parser.add_argument('--compress', type=float, help='compress_rate', default=1)
    parser.add_argument('--branch', type=float, help='num_conv_branches', default=3)
    args = parser.parse_args()
    return args

train_shape_dict = {
    'uci':[128, 9],
    'pamap2':[342, 36],
    'unimib':[151, 3],
    'wisdm':[200, 3],
    'usc':[512, 6]
}

category_dict = {
    'uci': 6,
    'pamap2': 12,
    'unimib': 17,
    'wisdm': 6,
    'usc': 12
}

if __name__ == "__main__":
    args = parse_args()
    setup_seed(7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, train_data_size, test_data_size = load_har_data(dataset_name=args.dataset, batch_size=args.batch)
    print('---------------------------------')
    print('dataset:{}'.format(args.dataset))

    if args.model == 'cnn':
        net = CNN(train_shape=train_shape_dict[args.dataset], category=category_dict[args.dataset], compress_rate=args.compress)
    elif args.model == 'mobilenet':
        net = MobileNet(train_shape=train_shape_dict[args.dataset], category=category_dict[args.dataset], compress_rate=args.compress)
    elif args.model == 'repmobile':
        net = RepMobile(num_classes=category_dict[args.dataset], compress_rate=args.compress, inference_mode=False, num_conv_branches=args.branch)

    net.to(device)
    print('model:{}'.format(args.model))
    # print(net)
    if args.model == 'repmobile':
        net.inference_mode = True
    count_FLOPS(net, data_shape=train_shape_dict[args.dataset])
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.001)
    list_test_acc = []
    for i in range(args.epoch):
        total_train_acc = 0
        total_test_acc = 0
        print("第{}轮训练开始".format(i + 1))
        net.train()
        if args.model == 'repmobile':
            net.inference_mode = False
        for step, (b_x, b_y) in enumerate(train_loader, 1):
            b_x = b_x.to(device).float()
            outputs = net(b_x)
            b_y = b_y.to(device).long()
            loss = loss_function(outputs, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_train = (outputs.argmax(1) == b_y).sum()  # 计算准确率仅用最后分类器
            total_train_acc = total_train_acc + acc_train

        train_acc = total_train_acc / train_data_size
        print("整体训练集上的正确率:{}".format(train_acc))
        net.eval()
        if args.model == 'repmobile':
            net = reparameterize_model(net)
            net.inference_mode = True
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(test_loader):
                b_x = b_x.to(device).float()
                outputs = net(b_x)
                b_y = b_y.to(device).long()
                acc_test = (outputs.argmax(1) == b_y).sum()
                total_test_acc = total_test_acc + acc_test
                test_acc = (total_test_acc / test_data_size)
        print("整体测试集上的正确率：{}".format(total_test_acc / test_data_size))
        x = test_acc.item()
        list_test_acc.append(x)
    avg_acc = sum(list_test_acc[-10:]) / 10
    print("最后10个epoch平均acc为{}".format(avg_acc))
    save_dir = args.dataset
    name = f'{args.model}_model.pth'
    save_path = os.path.join(save_dir, name)
    torch.save(net.state_dict(), save_path)
