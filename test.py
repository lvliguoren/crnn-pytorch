import torch
from dataset.ImageSet import CustomData
from torch.utils.data import DataLoader


with open('dataset/txt/char_std_5990.txt', 'rb') as file:
    char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}

test_data = CustomData('E:/TEST/Synthetic Chinese String Dataset/images', char_dict=char_dict, is_train=False)
test_loader = DataLoader(dataset=test_data,
                          batch_size=128,
                          shuffle=True,)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def val():
    model = torch.load('model/mycrnn.pth').to(device)

    model.eval()
    n_correct = 0
    n_total = 0
    for step, (batch_x, batch_y) in enumerate(test_loader):
        tensor_x = batch_x.to(device)
        # y = []
        # for row in batch_y:
        #     y.append([int(i) for i in row.split(',')])
        # tensor_y = torch.IntTensor(y)

        pred = model(tensor_x)
        _, pred_y = pred.max(2)
        pred_y = pred_y.permute(1, 0).cpu()
        p_y = []
        for row in pred_y:
            p_y.append(','.join([str(i.numpy()) for i in row if i > 0]))

        for text, label in zip(p_y, batch_y):
            if text == label:
                n_correct += 1

        n_total += len(batch_y)

    print('accuray:{}'.format(n_correct/n_total))


if __name__ == '__main__':
    val()
