import torch
import torch.utils.data as Data
import numpy as np
import random

from thop import profile


def load_har_data(dataset_name, batch_size=64):
    # 数据文件路径
    train_x_list = f'{dataset_name}/train_x.npy'
    train_y_list = f'{dataset_name}/train_y.npy'
    test_x_list = f'{dataset_name}/test_x.npy'
    test_y_list = f'{dataset_name}/test_y.npy'

    # 读取数据的类
    class HAR(Data.Dataset):
        def __init__(self, filename_x, filename_y):
            self.filename_x = filename_x
            self.filename_y = filename_y

        def HAR_data(self):
            data_x = np.load(self.filename_x)
            print(f'Loaded data_x shape: {data_x.shape}')
            data_x = data_x.reshape(-1, 1, data_x.shape[-2], data_x.shape[-1])  # 修改x的维度
            print(f'Reshaped data_x shape: {data_x.shape}')
            data_y = np.load(self.filename_y)
            print(f'Loaded data_y shape: {data_y.shape}')
            torch_dataset = Data.TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
            return torch_dataset

    # 加载训练数据和测试数据
    data_train = HAR(train_x_list, train_y_list)
    torch_train_dataset = data_train.HAR_data()
    train_data_size = len(torch_train_dataset)
    print("Train data loading completed")

    data_test = HAR(test_x_list, test_y_list)
    torch_test_dataset = data_test.HAR_data()
    test_data_size = len(torch_test_dataset)
    print("Test data loading completed")

    # 创建DataLoader
    train_loader = Data.DataLoader(dataset=torch_train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0)

    test_loader = Data.DataLoader(dataset=torch_test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0)

    return train_loader, test_loader, train_data_size, test_data_size
# train_loader, test_loader = load_har_data('UCI', batch_size=64)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_FLOPS(model, data_shape):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input = torch.randn(1, 1, data_shape[0], data_shape[1]).to(device)
    flops, params = profile(model, inputs=(input,))
    print("FLOPS：{}".format(flops))
    print("Para：{}".format(params))
    print('---------------------------------')
    return flops, params
