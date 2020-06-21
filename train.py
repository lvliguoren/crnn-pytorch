import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from model.crnn import CRNN


data_loader = DataLoader()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(num_class=3755).to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
loss_func = nn.CTCLoss()

def train():
    model.train()

    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        pred_length = torch.Tensor([i.size(0) for i in batch_y])
        target = batch_y.view(1, -1)
        target_length = torch.Tensor(pred_length.size())
        loss = loss_func(pred, target, pred_length, target_length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            status = "step: {}, loss: {},".format(step, loss.item())
            print(status)

    torch.save(model, 'crnn.pth')

if __name__ == '__main__':
    train()