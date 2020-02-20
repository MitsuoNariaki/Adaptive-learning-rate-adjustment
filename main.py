import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import tensorflow as tf
import torch.utils.data as data
import cnn
import dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

torch.manual_seed(1)

batch_size = 100
num_classes = 10
epochs = 12
global_step = 0

lr_list = [1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7,5e-8]
N = len(lr_list)
M = 4*np.log(N)
beta = 1/(1+np.sqrt(2*np.log(N)/M))
w = np.array([1/N for i in range(N)])
#w_map = np.array([1/N for i in range(N)])
#Loss_Matrix = np.zeros(N)
LoadModelName0 = 'weight.pth'



def WAA(loss_list,w):
    
    selected = np.random.choice(len(loss_list), p=w)
    
    w_Mom = sum(w*pow(beta,loss_list))
    
    for i in range(N):
        w_Child = w[i]*pow(beta,loss_list[i])
        w[i] = w_Child/w_Mom
    
    return selected,w



def cal_Regret(Loss_Matrix,P_loss_list):
    Record = np.sum(Loss_Matrix,axis = 0)
    Best_Ex_indent = np.argmin(Record)
    Best_Ex_loss = []
    b = 0
    for  i in Loss_Matrix[1:,Best_Ex_indent]:
        b = b + i
        Best_Ex_loss.append(b)
    Regret = np.array(P_loss_list) - np.array(Best_Ex_loss)
    
    return Regret


 
def train(epoch,LR,i):
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
 
    print("\n--- Epoch : %2d _%2d ---" % (epoch,i+1))
        
    if epoch < 4:
        dataloader_train = dataloader_train1
        steps = len(ds_train1)//batch_size
    elif 4 <= epoch < 8:
        dataloader_train = dataloader_train2
        steps = len(ds_train2)//batch_size
    else:
        dataloader_train = dataloader_train3
        steps = len(ds_train3)//batch_size

    for step, (images, labels) in enumerate(dataloader_train, 1):
        global global_step
        global_step += 1

        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, epochs, step, steps, loss.item()))
            #writer.add_scalar('train/train_loss', loss.item() , global_step)

            
def test(epoch,i):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        if epoch < 4:
            dataloader_test = dataloader_test1
        elif 4 <= epoch < 8:
            dataloader_test = dataloader_test2
        else:
            dataloader_test = dataloader_test3
        
        for (images, labels) in dataloader_test:
            images, labels = images.to(device), labels.to(device)
 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
 
    print("Test Acc : %.4f" % (correct/total))
    print("Test Err : %.4f" % (1-correct/total))

    #writer.add_scalar('eval/val_acc', correct*100/total, epoch)
    return 1-correct/total,correct/total



model = cnn.Cifar10Model().to(device)
criterion = nn.CrossEntropyLoss()

ds_train1,ds_train2,ds_train3 = dataset.changingtraindata(batch_size)
dataloader_train1 = data.DataLoader(dataset=ds_train1, batch_size=batch_size, shuffle=True) 
dataloader_train2 = data.DataLoader(dataset=ds_train2, batch_size=batch_size, shuffle=True) 
dataloader_train3 = data.DataLoader(dataset=ds_train3, batch_size=batch_size, shuffle=True)

ds_test1,ds_test2,ds_test3 = dataset.changingtestdata(batch_size)
dataloader_test1 = data.DataLoader(dataset=ds_test1, batch_size=batch_size, shuffle=False)
dataloader_test2 = data.DataLoader(dataset=ds_test2, batch_size=batch_size, shuffle=False)
dataloader_test3 = data.DataLoader(dataset=ds_test3, batch_size=batch_size, shuffle=False)

P_acc_list = []
for epoch in range(1, epochs+1):
    loss_list = []
    acc_list = []
        
    for i,LR in enumerate(lr_list):
        if epoch!=1:
            LoadModelName = 'weight'+str(selected)+'.pth'
            param = torch.load(LoadModelName)
            model.load_state_dict(param)
            
        else:  
            param0 = torch.load(LoadModelName0)
            model.load_state_dict(param0)
        
        train(epoch,LR,i)
        SaveModelName = 'weight'+str(i)+'.pth'
        torch.save(model.state_dict(), SaveModelName)
        l,a = test(epoch,i)
        acc_list.append(a)
        loss_list.append(l)
       
    selected,w = WAA(loss_list,w)
    
    P_acc_list.append(acc_list[selected])
    
    #==change in weight==    
    #w_map = np.vstack((w_map,w))
    #=======
    
#===calculate regret====    

    #Loss_Matrix = np.vstack((Loss_Matrix,loss_list))
    #P_loss = P_loss + loss_list[selected]
    #P_loss_list.append(P_loss)
    

#Regret = cal_Regret(Loss_Matrix,P_loss_list)

#======================

writer.close()



