import tensorflow as tf
import numpy as np
import torch
import torch.utils.data as data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
 
x_train = np.moveaxis(x_train, [3, 1, 2], [1, 2, 3]).astype('float32')
x_test = np.moveaxis(x_test, [3, 1, 2], [1, 2, 3]).astype('float32')
 
x_train /= 255
x_test /= 255
 
y_train = y_train.reshape(-1).astype('long')
y_test = y_test.reshape(-1).astype('long')




def changingtraindata(batch_size):
	
    
    x_train1 = []
    y_train1 = []

    x_train2 = []
    y_train2 = []

    x_train3 = []
    y_train3 = []


    for i in np.array(np.where(y_train[:15000] > 4))[0]:
        x_train1.append(x_train[:15000][i])
        y_train1.append(y_train[:15000][i])
    x_train1 = np.array(x_train1)
    y_train1 = np.array(y_train1)

    for i in np.array(np.where(y_train[15000:30000] < 5))[0]:
        x_train2.append(x_train[15000:30000][i])
        y_train2.append(y_train[15000:30000][i])
    x_train2 = np.array(x_train2)
    y_train2 = np.array(y_train2)

    for i in np.array(np.where(y_train[30000:] > 4))[0]:
        x_train3.append(x_train[30000:][i])
        y_train3.append(y_train[30000:][i])
    x_train3 = np.array(x_train3)
    y_train3 = np.array(y_train3)
    
    ds_train1 = data.TensorDataset(torch.from_numpy(x_train1), torch.from_numpy(y_train1))
    ds_train2 = data.TensorDataset(torch.from_numpy(x_train2), torch.from_numpy(y_train2))
    ds_train3 = data.TensorDataset(torch.from_numpy(x_train3), torch.from_numpy(y_train3))
    
    #dataloader_train1 = data.DataLoader(dataset=ds_train1, batch_size=batch_size, shuffle=True) 
    #dataloader_train2 = data.DataLoader(dataset=ds_train2, batch_size=batch_size, shuffle=True) 
    #dataloader_train3 = data.DataLoader(dataset=ds_train3, batch_size=batch_size, shuffle=True)
    
    #return dataloader_train1,dataloader_train2,dataloader_train3
    return ds_train1,ds_train2,ds_train3

def changingtestdata(batch_size):
    
    x_test1 = []
    y_test1 = []

    x_test2 = []
    y_test2 = []

    x_test3 = []
    y_test3 = []

    for i in np.array(np.where(y_test[:3000] > 4))[0]:
        x_test1.append(x_test[:3000][i])
        y_test1.append(y_test[:3000][i])
    x_test1 = np.array(x_test1)
    y_test1 = np.array(y_test1)


    for i in np.array(np.where(y_test[3000:6000] < 5))[0]:
        x_test2.append(x_test[3000:6000][i])
        y_test2.append(y_test[3000:6000][i])
    x_test2 = np.array(x_test2)
    y_test2 = np.array(y_test2)

    for i in np.array(np.where(y_test[6000:] > 4))[0]:
        x_test3.append(x_test[6000:][i])
        y_test3.append(y_test[6000:][i])
    x_test3 = np.array(x_test3)
    y_test3 = np.array(y_test3)
    
    ds_test1  = data.TensorDataset(torch.from_numpy(x_test1), torch.from_numpy(y_test1))
    ds_test2  = data.TensorDataset(torch.from_numpy(x_test2), torch.from_numpy(y_test2))
    ds_test3  = data.TensorDataset(torch.from_numpy(x_test3), torch.from_numpy(y_test3))

    #dataloader_test1 = data.DataLoader(dataset=ds_test1, batch_size=batch_size, shuffle=False)
    #dataloader_test2 = data.DataLoader(dataset=ds_test2, batch_size=batch_size, shuffle=False)
    #dataloader_test3 = data.DataLoader(dataset=ds_test3, batch_size=batch_size, shuffle=False)

    #return dataloader_test1,dataloader_test2,dataloader_test3
    return ds_test1,ds_test2,ds_test3






