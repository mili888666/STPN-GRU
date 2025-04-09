# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:36:04 2022

@author: AA
"""

import torch


import util
import argparse
import random
import copy
import torch.optim as optim
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from matplotlib import pyplot as plt
from baseline_methods import test_error, StandardScaler
from model import STPN

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='US',help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='train/val/test ratio', type=float)#训练、验证和测试集的比例（70%、10%、20%）
parser.add_argument('--h_layers',type=int,default=2,help='number of hidden layer')
parser.add_argument('--in_channels',type=int,default=1,help='input variable')
parser.add_argument("--hidden_channels", nargs="+", default=[128, 64, 32], help='hidden layer dimension', type=int)
parser.add_argument('--out_channels',type=int,default=1,help='output variable')
parser.add_argument('--emb_size',type=int,default=16,help='time embedding size')#--emb_size：时间嵌入维度（时间特征的编码大小）。
parser.add_argument('--dropout',type=float,default=0,help='dropout rate')#--dropout：用于防止过拟合（默认 0，即不使用 Dropout）。
parser.add_argument('--wemb_size',type=int,default=16,help='covairate embedding size')#--wemb_size：外部变量（如天气数据）的嵌入维度。
parser.add_argument('--time_d',type=int,default=4,help='normalizing factor for self-attention model')#--time_d：自注意力模型的归一化因子（用于缩放 QK^T）。
parser.add_argument('--heads',type=int,default=4,help='number of attention heads')#自注意力的多头注意力数量
parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
parser.add_argument('--order',type=int,default=2,help='order of diffusion convolution')
parser.add_argument('--num_weather',type=int,default=14,help='number of weather condition')
parser.add_argument('--use_se', type=str, default=True,help="use SE block")
parser.add_argument('--use_cov', type=str, default=True,help="use Covariate")
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')#学习率衰减
parser.add_argument('--lr', type=float, default=0.001, help='learning rate ')#学习率
parser.add_argument('--in_len',type=int,default=48,help='input time series length')#输入的时间序列长度
parser.add_argument('--out_len',type=int,default=6,help='output time series length')#输出的时间序列长度
parser.add_argument('--batch',type=int,default=32,help='training batch size')#批量大小（每次训练 32 个样本）。
parser.add_argument('--episode',type=int,default=100,help='training episodes')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')#--period：用于时间嵌入（周期性时间信息）。

args = parser.parse_args()

# 在 main() 函数开头添加以下函数
def set_seed(seed=42):
    # Python内置随机数
    random.seed(seed)

    # NumPy随机数
    np.random.seed(seed)

    # PyTorch随机数 (CPU)
    torch.manual_seed(seed)

    # PyTorch随机数 (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时设置所有

    # 确保CuDNN的确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 操作系统环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main():
    set_seed(42)  # 设置全局种子为42
    """数据加载"""
    device = torch.device(args.device)
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)

    # 添加维度验证
    print("关键数据维度检查:")
    print("训练数据维度:", training_data.shape)  # 应为 (30, T, 1)
    print("邻接矩阵列表维度:", [a.shape for a in adj])  # 应全部为 (30,30)

    """初始化STPN模型,定义Adam优化器"""
    #adj：空间邻接矩阵、training_data, val_data, test_data：训练、验证、测试数据。training_w, val_w, test_w：训练、验证、测试天气数据。
    model = STPN(args.h_layers, args.in_channels, args.hidden_channels, args.out_channels, args.emb_size, 
                 args.dropout, args.wemb_size, args.time_d, args.heads, args.support_len,
                 args.order, args.num_weather, args.use_se, args.use_cov).to(device)
    # 将动态邻接矩阵参数与其他参数分离
    dynamic_params = []
    other_params = []
    for name, param in model.named_parameters():
        # 筛选动态矩阵相关参数（包括动态邻接矩阵和融合权重）
        if "dynamic_adj" in name or "fusion_weights" in name:
            dynamic_params.append(param)
        else:
            other_params.append(param)

    # 定义分层学习率（动态参数更低学习率）
    optimizer = optim.Adam([
        {"params": dynamic_params, "lr": args.lr * 0.1},  # 动态参数学习率设为其他参数的1/10
        {"params": other_params, "lr": args.lr}  # 其他参数保持原学习率
    ], weight_decay=args.decay)

    # 验证参数分组是否正确（调试时使用）
    print(f"动态参数数量: {len(dynamic_params)}, 其他参数数量: {len(other_params)}")

    #隐藏层数量（h_layers）、输入变量数量（in_channels=2，如到达延误和离港延误）、注意力头数（heads=4）。扩散卷积的阶数（order=2）。天气条件（num_weather=8）。
    supports = [torch.tensor(i).to(device) for i in adj]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    """标准化数据(均值为0，标准差为1)，填充NaN为0，防止出错"""
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(), training_data[~np.isnan(training_data)].std())
    #计算均值和标准差，用于标准化（StandardScaler）。
    training_data = scaler.transform(training_data)
    #scaler.transform(training_data)：标准化数据，使其均值为 0，方差为 1。
    training_data[np.isnan(training_data)] = 0
    #training_data[np.isnan(training_data)] = 0：填充缺失值为 0，避免 NaN 影响模型训练。

    """模型训练"""
    MAE_list = []
    R2_list = []
    #作用：创建一个空列表 MAE_list，用于存储每个训练 episode（训练轮数）的均值绝对误差（MAE）。
    batch_index = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    #batch_index：获取训练集可用的时间索引。training_data.shape[1]：表示时间维度（即有多少个时间步）。
    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))
    #val_index：获取验证集可用的时间索引。选定的时间窗口不会超出数据范围。有足够的 in_len 作为输入，也有足够的 out_len 作为预测目标。

    label = []
    #label：存储所有验证数据的真实延误数据（未来 out_len 个时间步）。
    for i in range(len(val_index)):
        label.append(np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis = 0))
    label = np.concatenate(label)

    #np.expand_dims(..., axis=0)：增加一个维度，使得数据格式统一。
    #label.append(...)：将每个时间窗口的目标数据加入 label 列表
    #np.concatenate(label)：将所有数据合并成一个完整的 label 张量。

    print("start training...",flush=True)
    
    for ep in range(1,1+args.episode):#外部循环：进行 args.episode 轮训练（即 ep 从 1 到 args.episode）。
        random.shuffle(batch_index)#打乱 batch_index，确保数据顺序随机化，防止模型学习固定模式，提高泛化能力。
        for j in range(len(batch_index) // args.batch - 1):
        #len(batch_index)：表示所有可用的时间窗口数,args.batch：批量大小（32）。
        #len(batch_index) // args.batch：计算 总共有多少个完整批次。
            trainx = []#存储输入的时间序列数据（过去 in_len 个时间步）。
            trainy = []#存储输出（真实的 out_len 目标）。
            trainti = []#存储时间嵌入信息（输入的时间步）。
            trainto = []#存储时间嵌入信息（输出的时间步）。
            trainw = []#存储天气数据（外部变量）。
            for k in range(args.batch):
                trainx.append(np.expand_dims(training_data[:, batch_index[j * args.batch +k]: batch_index[j * args.batch +k] + args.in_len, :], axis = 0))
                #输入的时间序列数据（过去in_len个时间步）
                trainy.append(np.expand_dims(training_data[:, batch_index[j * args.batch +k] + args.in_len:batch_index[j * args.batch +k] + args.in_len + args.out_len, :], axis = 0))
                #真实输出数据(out_len个实践部)
                # 修改后（添加batch维度）：
                trainw.append(np.expand_dims(training_w[:, batch_index[j * args.batch + k]: batch_index[j * args.batch + k] + args.in_len,:],axis=0))
                #存储天气数据（外部变量）
                trainti.append((np.arange(batch_index[j * args.batch +k], batch_index[j * args.batch +k] + args.in_len) % args.period) * np.ones([1, args.in_len])/(args.period - 1))
                #输入时间步的时间嵌入信息
                trainto.append((np.arange(batch_index[j * args.batch +k] + args.in_len, batch_index[j * args.batch +k] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len])/(args.period - 1))
                #输出时间步的时间嵌入信息
                #%args.period能够确保所有的时间步在[0,rags.period-1]
                #np.ones([1, args.in_len]) 生成一个形状为 (1, in_len) 的全 1 数组，使得原来的一维数组（12,）转换为（1,12）
                #/(args.period - 1)归一化操作 使得所有步长都在[0,1]内
            trainx = np.concatenate(trainx)#ndarray:(32,30,12,1)(batch,N,in_len,C)
            trainti = np.concatenate(trainti)#(period,in_len)
            trainto = np.concatenate(trainto)#(period,out_len)
            trainy = np.concatenate(trainy)#(batch,N,out_len,C)
            trainw = np.concatenate(trainw)#(batch,N,in_len,C)
            #np.concatenate(...) 将 batch 内所有样本拼接，形成最终批次数据。

            # 新代码（新天气数据，14维特征）
            trainw = torch.FloatTensor(trainw).to(device)  # 注意改为 FloatTensor

            trainx = torch.Tensor(trainx).to(device)
            trainx= trainx.permute(0, 3, 1, 2)
            #训练数据 trainx 原始形状 (batch, N, in_len, C)。
            #经过 permute(0, 3, 1, 2) 变为 (batch, C, N, in_len)，适配 STPN 模型输入。

            trainy = torch.Tensor(trainy).to(device)
            trainy = trainy.permute(0, 3, 1, 2)

            trainti = torch.Tensor(trainti).to(device)
            trainto = torch.Tensor(trainto).to(device)

            model.train()#model.train()：设置模型为训练模式（启用 dropout）。
            optimizer.zero_grad()#清除梯度，避免梯度累积。

            output = model(trainx, trainti, supports, trainto, trainw)#model(...)：前向传播，计算 output（模型预测值）。
            loss = util.masked_rmse(output, trainy, 0.0)#util.masked_rmse(output, trainy, 0.0) 计算均方根误差（RMSE）。
            loss.backward()#计算梯度。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)#梯度裁剪，防止梯度爆炸（最大范数 3）。
            optimizer.step()#更新模型参数。
            
        outputs = []#存储所有预测值。
        model.eval()#切换到评估模式（关闭 dropout）。
        #将模型切换为评估模式，高速模型不要更新参数（参数不会改变）、关闭dropout(防止评估时随机丢弃神经元)、 关闭 Batch Normalization（BN）动量更新（防止均值/方差变化）。
        for i in range(len(val_index)):#val_index 存储验证集的所有时间窗口的起始索引。
            testx = np.expand_dims(val_data[:, val_index[i]: val_index[i] + args.in_len, :], axis = 0)
            #val_data.shape = (N,in_len,C),通过np.expand_dims增加了batch维度，变成了（1，N，in_len,C）,符合神经网络的输入格式。
            testx = scaler.transform(testx)
            #测试数据textx通过scaler进行标准化
            testx[np.isnan(testx)] = 0
            #testx[np.isnan(testx)] = 0

            testw = np.expand_dims(val_w[:, val_index[i]:val_index[i]+args.in_len, :], axis=0)  # 保持四维结构(batch, N, T, 14)
            #val_w 是天气数据（shape 为 (N, T)）。取出 in_len 时间步的天气数据，并增加 batch 维度
            testw = torch.FloatTensor(testw).to(device)  # 改为FloatTensor
            #转换为 PyTorch LongTensor，并移动到 device（CPU/GPU）。

            testti = (np.arange(int(training_data.shape[1])+val_index[i], int(training_data.shape[1])+val_index[i]+ args.in_len) % args.period) * np.ones([1, args.in_len])/(args.period - 1)
            #testti 代表输入时间步的时间编码
            testto = (np.arange(int(training_data.shape[1])+val_index[i] + args.in_len, int(training_data.shape[1])+val_index[i] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len])/(args.period - 1)
            #testto 代表输出时间步的时间编码（未来 out_len 个时间步）。
            testx = torch.Tensor(testx).to(device)
            testx= testx.permute(0, 3, 1, 2)

            testti = torch.Tensor(testti).to(device)
            testto = torch.Tensor(testto).to(device)

            output = model(testx, testti, supports, testto, testw)
            #model(...) 进行前向传播，得到预测结果 output。

            output = output.permute(0, 2, 3, 1)
            # output 原始形状：(1, C, N, out_len)。permute(0, 2, 3, 1) 调整回 (1, N, out_len, C)。

            output = output.detach().cpu().numpy()
            #.detach().cpu().numpy() 转换为 NumPy 数组，方便后续计算。

            output = scaler.inverse_transform(output)
            #scaler.inverse_transform(output) 还原标准化前的数值。

            outputs.append(output)
            #outputs.append(output) 存储所有时间窗口的预测结果。

        yhat = np.concatenate(outputs)
         
        amae = []
        ar2 = []
        armse = []
        for i in range(args.out_len):
            metrics = test_error(yhat[:,:,i,:],label[:,:,i,:])
            #yhat[:,:,i,:]  # 取出第 i 个预测时间步的所有机场预测值
            #label[:,:,i,:]  # 取出第 i 个预测时间步的所有机场真实值
            amae.append(metrics[0])#amae 存储 12 个时间步的 MAE（误差越小越好）。
            ar2.append(metrics[2])#ar2 存储 12 个时间步的 R²（越接近 1 越好）。
            armse.append(metrics[1])#armse 存储 12 个时间步的 RMSE（误差越小越好）。
         
        log = 'On average over all horizons, Test MAE: {:.4f}, Test R2: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(ar2),np.mean(armse)))
     
        MAE_list.append(np.mean(amae))
        R2_list.append(np.mean(ar2))
        #MAE_list 是一个列表，存储每个训练轮次的平均 MAE（均值绝对误差）。


        # 定义导出为Excel的函数
        def export_predictions_to_excel(predictions, filename="best_model_predictions.xlsx"):
            # 预测值 predictions 是一个数组或列表，需要将其转换为 pandas DataFrame
            # 将预测值转化为 DataFrame 格式后导出到 Excel
            df = pd.DataFrame(predictions.reshape(-1, predictions.shape[-1]))
            df.to_excel(filename, index=False)
            print(f"Predictions exported to {filename}")

        # 在最优模型部分修改：
        if np.mean(amae) == min(MAE_list):
            best_model = copy.deepcopy(model.state_dict())
            best_predictions = yhat  # 保存此时的预测值
            export_predictions_to_excel(best_predictions)  # 导出预测值到Excel

    model.load_state_dict(best_model)
    torch.save(model, "spdpn" + args.data +".pth")
    # 绘制验证集的真实值和预测值对比曲线
    plt.figure(figsize=(10, 5))

    # 绘制第一个机场的真实值曲线
    plt.plot(label[:, 0, 0, 0].reshape(-1), color='blue', linestyle='-', marker='o', label='True')

    # 绘制第一个机场的预测值曲线
    plt.plot(best_predictions[:, 0, 0, 0].reshape(-1), color='red', linestyle='--', marker='x', label='Predicted')

    # 设置标题和标签
    plt.title('Prediction vs Real Values for Best Model on Validation Dataset')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.grid(True)
    # 绘制 MAE 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.episode + 1), MAE_list, marker='o', linestyle='-', color='b')
    plt.title('Validation MAE over Training Episodes')
    plt.xlabel('Episode')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.savefig('training_mae_curve.png')
    plt.show()
    model.load_state_dict(best_model)
    torch.save(model, "spdpn" + args.data +".pth")
    # 绘制 R2 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.episode + 1), R2_list, marker='o', linestyle='-', color='b')
    plt.title('Validation R2 over Training Episodes')
    plt.xlabel('Episode')
    plt.ylabel('R2')
    plt.grid(True)
    plt.savefig('training_R2_curve.png')
    plt.show()
    model.load_state_dict(best_model)
    torch.save(model, "spdpn" + args.data + ".pth")
if __name__ == "__main__":   
    main()


