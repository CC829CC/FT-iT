import random
import time
import pandas as pd
import numpy as np
from utils.timefeatures import time_features
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from iTransformer.core import fretime_iTransformer
from sklearn.metrics import mean_absolute_percentage_error

seed = random.randint(0, 2 ** 31 - 1)
print('seed', seed)
UseWTConv = False
UseTCN = True
UseFrequency = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
file_path = 'data.csv'
model_road = 'model_saveload'
name = 'data_name'
window = 'input series length'
length_size = 1
label_len = int(window / 2)
epochs = 30
batch_size = 32
df_raw = pd.read_csv(file_path)
df_stamp = df_raw[['trade_date']]
df_stamp['date'] = pd.to_datetime(df_stamp.trade_date)
data_stamp = time_features(df_stamp, timeenc=1, freq='d')
print(data_stamp)
data = df_raw.iloc[:, 1:]
data_target = df_raw['close']
data_target_index = df_raw.columns.get_loc('close')
data_dim = len(data.iloc[1, :])
scaler = preprocessing.StandardScaler()
if data_dim == 1:
    data_inverse = scaler.fit_transform(np.array(data).reshape(-1, 1))
else:
    data_inverse = scaler.fit_transform(np.array(data))
data = data_inverse
data_length = len(data)
train_set = 0.7
val_set = 0.9
# 获取训练集
data_train = data[:int(train_set * data_length), :]
# 获取训练集对应的时间特征
data_train_mark = data_stamp[:int(train_set * data_length), :]
# 获取验证集
data_val = data[int(train_set * data_length):int(val_set * data_length), :]
# 获取验证集对应的时间特征
data_val_mark = data_stamp[int(train_set * data_length):int(val_set * data_length), :]
# 获取测试集
data_test = data[int(val_set * data_length):, :]
# 获取训练集对应的时间特征
data_test_mark = data_stamp[int(val_set * data_length):, :]
n_feature = data_dim


def data_loader(window, length_size, batch_size, data, data_mark, shuffle):
    seq_len = window  # 以规定时间窗作为预测序列的长度
    sequence_length = seq_len + length_size  # 序列长度，也就是输入序列的长度+预测序列的长度
    result = []
    result_mark = []
    # 分割数据，通过滑动窗口的方式生成输入序列和目标序列
    for index in range(len(data) - sequence_length + 1):
        result.append(data[index: index + sequence_length])  # 第i行到i+sequence_length
        result_mark.append(data_mark[index: index + sequence_length])
    result = np.array(result)  # 得到样本，样本形式为sequence_length*特征
    result_mark = np.array(result_mark)
    x_train = result[:, :-length_size]  # 训练集特征数据
    print('x_train shape:', x_train.shape)
    x_train_mark = result_mark[:, :-length_size]
    print('x_train_mark shape:', x_train_mark.shape)
    y_train = result[:, -(length_size + int(window / 2)):]  # 训练集目标数据
    print('y_train shape:', y_train.shape)
    y_train_mark = result_mark[:, -(length_size + int(window / 2)):]
    print('y_train_mark shape:', y_train_mark.shape)

    x_train, y_train = torch.tensor(x_train).to(torch.float32), torch.tensor(y_train).to(
        torch.float32)  # 将数据转变为tensor张量
    x_train_mark, y_train_mark = torch.tensor(x_train_mark).to(torch.float32), torch.tensor(y_train_mark).to(
        torch.float32)  # 将数据转变为tensor张量
    ds = torch.utils.data.TensorDataset(x_train, y_train, x_train_mark, y_train_mark)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                             drop_last=False)  # 对训练集数据进行打包，每32个数据进行打包一次，组后一组不足32的自动打包
    return dataloader, x_train, y_train, x_train_mark, y_train_mark


dataloader_train, x_train, y_train, x_train_mark, y_train_mark = data_loader(window, length_size, batch_size,
                                                                             data_train, data_train_mark, True)
dataloader_val, x_val, y_val, x_val_mark, y_val_mark = data_loader(window, length_size, batch_size, data_val,
                                                                   data_val_mark, False)
dataloader_test, x_test, y_test, x_test_mark, y_test_mark = data_loader(window, length_size, batch_size, data_test,
                                                                        data_test_mark, False)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


class Config:
    def __init__(self):
        # self.features = "M"
        self.hist_len = window  #
        self.pred_len = length_size  #
        self.e_layers = 'num of Encoder'
        self.enc_in = n_feature
        self.d_model = 'important para of model'
        self.d_ff = 'important para of model'
        self.kernel_size = 2
        self.wt_levels = 3
        self.channel = 'out channel'
        self.dropout = 'dropout rate'
        self.factor = 3
        self.n_heads = 'num of head'
        self.output_attention = 0
        self.activation = 'activation fuction(non-linear)'
        self.patience = 'early stopping'
        self.lr = 'learning rate'
        self.UseWTConv = UseWTConv
        self.UseTCN = UseTCN
        self.UseFre = UseFrequency


config = Config()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False  ###早停标志
        self.val_loss_min = np.Inf  ###这一行将实例变量val_loss_min设置为正无穷
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def fretime_iTransformer_train(config):
    net = fretime_iTransformer(config).to(device)
    criterion = nn.MSELoss().to(device)  # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=1e-5)  # 优化算法和学习率
    best_model_path = model_road + 'FT-iTransformer.pt'
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    """
    模型训练过程
    """
    LOSS_train = []
    LOSS_val = []
    for epoch in range(epochs):  # 10
        net.train()
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(dataloader_train):
            datapoints, datapoints_mark, labels_mark, labels = datapoints.to(device), datapoints_mark.to(
                device), labels_mark.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = net(datapoints, datapoints_mark)
            loss = criterion(preds, labels[:, -length_size:, :])
            LOSS_train.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()

        net.eval()  # 设置为评估模式
        with torch.no_grad():

            for j, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(dataloader_val):
                datapoints, datapoints_mark, labels_mark, labels = datapoints.to(device), datapoints_mark.to(
                    device), labels_mark.to(device), labels.to(device)
                optimizer.zero_grad()
                preds = net(datapoints, datapoints_mark)
                loss = criterion(preds, labels[:, -length_size:, :])
                LOSS_val.append(loss.detach().cpu().numpy())
        print("epoch: {} Train-loss: {:.6f} ===== Val-loss: {:.6f}".format(str(epoch), np.mean(np.array(LOSS_train)),
                                                                           np.mean(np.array(LOSS_val))))
        early_stopping(np.mean(np.array(LOSS_val)), net, best_model_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        # adjust_learning_rate(optimizer, epoch + 1, config.lr, lradj='type1')

    return net


def fretime_iTransformer_test(config):
    net = fretime_iTransformer(config).to(device)
    net.load_state_dict(torch.load(model_road + 'FT-iTransformer.pt'))  # 加载训练好的模型
    net.eval()

    preds = []
    trues = []
    for j, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(dataloader_test):
        datapoints, datapoints_mark, labels_mark, labels = datapoints.to(device), datapoints_mark.to(
            device), labels_mark.to(device), labels.to(device)
        pred = net(datapoints, datapoints_mark)
        pred = pred.detach().cpu()
        true = labels[:, -length_size:, :].detach().cpu()
        preds.append(pred)
        trues.append(true)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    print("Shape of pred before adjustment:", preds.shape)
    print("Shape of true before adjustment:", trues.shape)
    # 可能需要调整pred和true的维度，使其变为二维数组
    true = trues[:, :, data_target_index - 1]
    pred = preds[:, :, data_target_index - 1]  # 假设需要将pred调整为二维数组，去掉最后一维
    # true =np.array(true)

    y_data_test_inverse = scaler.fit_transform(
        np.array(data_target).reshape(-1, 1))  # 这段代码是为了重新更新scaler，因为之前定义的scaler是是十六维，这里重新根据目标数据定义一下scaler
    pred_uninverse = scaler.inverse_transform(pred)  # 如果是多步预测， 选取最后一列
    true_uninverse = scaler.inverse_transform(true)

    return true_uninverse, pred_uninverse


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


if __name__ == "__main__":
    road = 'save_load'
    start = time.perf_counter()
    fretime_iTransformer_train(config)
    end = time.perf_counter()
    true, pred = fretime_iTransformer_test(config)
    date = df_raw['trade_date'][-len(pred):].values.reshape(-1, 1)
    df_true = pd.DataFrame(np.concatenate((date, true), axis=1),
                           columns=['date'] + ['time stamp {}'.format(i + 1) for i in range(length_size)])  # 预测结果保存
    df_pred = pd.DataFrame(np.concatenate((date, pred), axis=1),
                           columns=['date'] + ['time stamp {}'.format(i + 1) for i in
                                               range(length_size)])
    df_true.to_csv(road)
    df_pred.to_csv(road)
    y_test = true[:, -1]
    y_test_predict = pred[:, -1]
    R2 = r2_score(y_test, y_test_predict)
    MAE = mean_absolute_error(y_test_predict, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test_predict, y_test))
    MAPE = mean_absolute_percentage_error(y_test, y_test_predict)
    print('MAE: {:.4f}'.format(MAE))
    print('RMSE: {:.4f}'.format(RMSE))
    print('MAPE: {:.4f}'.format(MAPE))
    print('R²: {:.4f}'.format(R2))
    print(f"{end - start:.4f} s")
    savef = pd.DataFrame()
    savef['MAE'] = [str(MAE)]
    savef['RMSE'] = [str(RMSE)]
    savef['MAPE'] = [str(MAPE)]
    savef['R2'] = [str(R2)]
    savef.to_csv(road)
