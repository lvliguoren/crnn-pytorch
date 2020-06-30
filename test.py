import torch
import torch.nn as nn
from dataset.ImageSet import CustomData
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def deduplication(pred_y):
    p_y = []
    for row in pred_y:
        row_y = []
        for word_idx in range(len(row)):
            if word_idx > 0:
                if row[word_idx] > 0 and row[word_idx] != row[word_idx - 1]:
                    row_y.append(str(row[word_idx].item()))
            else:
                if row[word_idx] > 0:
                    row_y.append(str(row[word_idx].item()))
        p_y.append(','.join(row_y))
    return p_y


def val():
    with open('dataset/txt/char_std_5990.txt', 'rb') as file:
        char_dict = {char.strip().decode('gbk', 'ignore'): num for num, char in enumerate(file.readlines())}
    test_data = CustomData('E:/TEST/Synthetic Chinese String Dataset/images', char_dict=char_dict, is_train=False)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=32,
                             shuffle=True, )
    model = torch.load('model/mycrnn.pth').to(device)
    loss_func = nn.CTCLoss().to(device)

    model.eval()
    n_word_correct = 0
    n_word_total = 0
    n_seq_correct = 0
    n_seq_total = 0
    for step, (batch_x, batch_y) in enumerate(test_loader):
        # tensor_x = batch_x.to(device)
        # y = []
        # for row in batch_y:
        #     y.append([int(i) for i in row.split(',')])
        # tensor_y = torch.IntTensor(y)

        batch_x = batch_x.to(device)
        target = []
        target_length = []
        for row in batch_y:
            word_idxs = row.split(',')
            for w_idx in word_idxs:
                target.append(int(w_idx))
            target_length.append(len(word_idxs))

        pred = model(batch_x)  # W,N,C
        pred = pred.to(torch.float64).to(device)
        pred_pro = pred.permute(1, 0, 2)  # N,W,C
        pred_length = torch.IntTensor([i.size(0) for i in pred_pro])
        target = torch.IntTensor(target).to(device)
        target_length = torch.IntTensor(target_length)
        loss = loss_func(pred, target, pred_length, target_length)

        _, pred_y = pred.max(2)
        pred_y = pred_y.permute(1, 0)
        p_y = deduplication(pred_y)

        for row, r_label in zip(p_y, batch_y):
            if row == r_label:
                n_seq_correct += 1
            words = [i for i in row.split(',')]
            labels = [i for i in r_label.split(',')]
            for word, w_label in zip(words, labels):
                if word == w_label:
                    n_word_correct += 1
                n_word_total += 1

        n_seq_total += len(batch_y)
        print('step{},test loss{}, sequence accuracy:{}, word accuracy{}'.format(step,loss.item(), n_seq_correct / n_seq_total, n_word_correct / n_word_total))

    print('sequence accuracy:{}, word accuracy{}'.format(n_seq_correct/n_seq_total, n_word_correct/n_word_total))


def test():
    with open('dataset/txt/char_std_5990.txt', 'rb') as file:
        num_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
    img = CustomData.get_file('test/20437093_2690788297.jpg')
    img = img.to(device)
    model = torch.load('model/mycrnn.pth').to(device)
    pred = model(img)
    _, pred_y = pred.max(2)
    pred_y = pred_y.permute(1, 0)
    # pred_y = pred_y.view(-1)
    p_y = deduplication(pred_y)
    text = []
    for word_idx in p_y[0].split(','):
        text.append(num_dict[int(word_idx)])
    print(''.join(text))

if __name__ == '__main__':
    val()
    # test()
