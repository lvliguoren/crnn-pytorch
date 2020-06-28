import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from model.crnn import CRNN
from dataset.ImageSet import CustomData
from torch.utils.tensorboard import SummaryWriter


with open('dataset/txt/char_std_5990.txt', 'rb') as file:
    char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}

train_data = CustomData('E:/TEST/Synthetic Chinese String Dataset/images', char_dict=char_dict, is_train=True)
train_loader = DataLoader(dataset=train_data,
                          batch_size=32,
                          shuffle=True,)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train():
    model.train()

    with SummaryWriter() as writer:
        for step, (batch_x, batch_y) in enumerate(train_loader):
            tensor_x = batch_x.to(device)
            y = []
            for row in batch_y:
                y.append([int(i) for i in row.split(',')])
            tensor_y = torch.IntTensor(y)

            # W,N,C
            pred = model(tensor_x)
            pred = pred.to(torch.float64)
            pred = pred.to(device)
            pred_pro = pred.permute(1,0,2)
            pred_length = torch.IntTensor([i.size(0) for i in pred_pro])
            target = tensor_y.view(-1)
            target = target.to(device)
            target_length =torch.IntTensor([len(i) for i in tensor_y])
            loss = loss_func(pred, target, pred_length, target_length)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                status = "step: {}, loss: {}".format(step, loss.item())
                print(status)
                writer.add_scalar("train_loss", loss.item(), step)

            if step!=0 and step%200 == 0:
                torch.save(model, 'model/mycrnn.pth')
    torch.save(model, 'model/mycrnn.pth')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(len(char_dict)-1).to(device)
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
loss_func = nn.CTCLoss().to(device)

if __name__ == '__main__':
    train()