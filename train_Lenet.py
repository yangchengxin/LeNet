import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Lenet_backbone import Lenet
from torchvision import datasets, transforms


device = torch.device("cuda:0")

class Config():
    batch_size = 128
    epoch = 10
    alpha = 1e-3
    print_per_step = 100

class Trainset():
    def __init__(self):
        self.train, self.test = self.load_data()
        self.net = Lenet().to(device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=Config.alpha)

    def load_data(self):
        train_data = datasets.MNIST(root = './data/',
                                    train=True,
                                    transform = transforms.Compose(
                                        [transforms.Resize((32, 32)), transforms.ToTensor()]),
                                    download=True,)

        test_data = datasets.MNIST(root = './data/',
                                   train=False,
                                   transform = transforms.Compose(
                                       [transforms.Resize((32, 32)), transforms.ToTensor()]),
                                   )

        train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                       batch_size=Config.batch_size,
                                                       shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                      batch_size=Config.batch_size,
                                                      shuffle=False)
        return train_dataloader, test_dataloader

    def train_step(self):
        print("Training & Evaluating based on ycx_net......")
        file = './result/train_mnist.txt'
        fp = open(file, 'w', encoding='utf-8')
        fp.write('epoch\tbatch\tloss\taccuracy\n')
        for epoch in range(Config.epoch):
            print("Epoch {:3}.".format(epoch + 1))
            for batch_idx, (data, label) in enumerate(self.train):
                data, label = Variable(data.cuda()), Variable(label.cuda())
                self.optimizer.zero_grad()
                outputs = self.net(data)
                loss = self.loss_function(outputs, label)
                loss.backward()
                self.optimizer.step()
                if batch_idx % Config.print_per_step == 0:
                    _, predicted = torch.max(outputs, 1)
                    correct = 0
                    for _ in predicted == label:
                        if _:
                            correct += 1
                    accuracy = correct / Config.batch_size
                    msg = "Batch: {:5}, Loss: {:6.2f}, Accuracy: {:8.2%}."
                    print(msg.format(batch_idx, loss, accuracy))
                    fp.write('{}\t{}\t{}\t{}\n'.format(epoch, batch_idx, loss, accuracy))
        fp.close()
        test_loss = 0
        test_correct = 0
        for data, label in self.test:
            data, label = Variable(data.cuda()), Variable(label.cuda())
            outputs = self.net(data)
            loss = self.loss_function(outputs, label)
            test_loss += loss * Config.batch_size
            _, predicted = torch.max(outputs, 1)
            correct = 0
            for _ in predicted == label:
                if _:
                    correct += 1
            test_correct += correct
        accuracy = test_correct / len(self.test.dataset)
        loss = test_loss / len(self.test.dataset)
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))
        torch.save(self.net.state_dict(), './result/raw_train_mnist_model.pth')

if __name__ == "__main__":
    p = Trainset()
    p.train_step()
