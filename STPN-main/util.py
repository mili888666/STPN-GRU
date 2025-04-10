# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:42:10 2022

@author: AA
"""
import numpy as np
import scipy.sparse as sp
import torch


def asym_adj(adj):
    """邻接矩阵处理"""
    adj = sp.coo_matrix(adj)  # 转换为稀疏矩阵
    rowsum = np.array(adj.sum(1)).flatten()  # 计算每个节点的度（行求和）
    d_inv = np.power(rowsum, -1).flatten()  # 计算度矩阵的逆
    d_inv[np.isinf(d_inv)] = 0.  # 处理无穷值（防止除零）
    d_mat = sp.diags(d_inv)  # 构造对角矩阵 D^(-1)
    return d_mat.dot(adj).astype(np.float32).todense()  # 计算 D^(-1) * A

def load_data(data_name, ratio = [0.7, 0.1]):
    if data_name == 'US':
        adj_mx = np.load('../my_data/airport_adjacency_gaussian_filtered.npy')
        #机场网络邻接矩阵
        od_power = np.load('../my_data/airport_traffic_matrix.npy')
        # 交通流量矩阵
        od_power = od_power/(1.5*od_power.max())
        # 归一化
        # 验证维度
        print("邻接矩阵维度验证:", adj_mx.shape, od_power.shape)  # 应输出 (30,30) (30,30)

        # 保持对角化处理
        od_power = od_power / (1.5 * od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(30):
            od_power[i, i] = 1

        # 生成标准化邻接矩阵列表
        adj = [
            asym_adj(adj_mx),
            asym_adj(od_power),
            asym_adj(od_power.T)
        ]
        """asym_adj(adj_mx): 机场之间的基础邻接矩阵。
        asym_adj(od_power): 乘客流量的邻接矩阵（方向：出发 -> 到达）。
        asym_adj(od_power.T): 乘客流量的邻接矩阵（方向：到达 -> 出发）。"""
        data = np.load('../my_data/airport_resilience_3d.npy')
        wdata = np.load('../my_data/airport_weather_operations_3d.npy')
    if data_name == 'China':
        adj_mx = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//cdata//dist_mx.npy')
        od_power = np.load('C:/Users\lenovo\Desktop\神经网络学习\STPN-main\my_data/airport_weather_operations_3d.npy')
        od_power = od_power/(1.5*od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(50):
            od_power[i, i] = 1
        adj = [asym_adj(adj_mx), asym_adj(od_power), asym_adj(od_power.T)]
        data = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//cdata//delay.npy.baiduyun.p.downloading')
        data[data<-15] = -15
        wdata = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//cdata//weather_cn.npy.baiduyun.p.downloading')
    training_data = data[:, :int(ratio[0]*data.shape[1]) ,:]
    val_data = data[:,int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1]),:]
    test_data = data[:, int((ratio[0] + ratio[1])*data.shape[1]):, :]

    training_w = wdata[:, :int(ratio[0]*data.shape[1])]
    val_w = wdata[:,int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1])]
    test_w = wdata[:, int((ratio[0] + ratio[1])*data.shape[1]):]

    return adj, training_data, val_data, test_data, training_w, val_w, test_w

def masked_mse(preds, labels, null_val=np.nan):
    """损失函数定义"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_wmae(preds, labels, weights, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask * weights
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
        
