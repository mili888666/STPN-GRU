import numpy as np

u_delay = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//udata//udelay.npy.baiduyun.p.downloading')
print(u_delay)
print(u_delay.shape)
#(70,78912,2):70个机场、78912个时间步、2个特征（到达延误、离港延误）
#(70,78912,1)
print('----------')

u_weather = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//udata//weather2016_2021.npy.baiduyun.p.downloading')
print(u_weather)
print(u_weather.shape)
print(u_weather[0].shape)
print(u_weather[0,:])
print('--------')
#(70,78912)--->#(70,78912,14)
# 获取所有唯一值
unique_values = np.unique(u_weather)

# 输出唯一值集合
print(unique_values)


adj_china = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//cdata//dist_mx.npy')
print(adj_china)
print(adj_china.shape)
od_china = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//cdata//od_mx.npy')
print(od_china)
print(od_china.shape)

print('--------')

adj_us= np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//udata//adj_mx.npy')
print(adj_us)
print(adj_us.shape)
"""
od_us = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//udata\od_pair.npy')
print(od_us)
print(od_us.shape)

data = [[[1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]],
        [[1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]],
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9],
         [10, 11, 12]]
]
data = np.array(data)
print(data.shape)
np.save('data.npy',data)
od_power = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//udata\od_pair.npy')
print("乘客流量矩阵：",od_power)
print(od_power.shape)
adj_mx = np.load('C://Users//lenovo//Desktop//神经网络学习//STPN-main//delay_data//udata//adj_mx.npy')
print(adj_mx)
wdata = np.load('C:/Users\lenovo\Desktop\神经网络学习\STPN-main\my_data/airport_weather_operations_3d.npy')
print(wdata)
"""