import torch
import torch.nn as nn
import torch.nn.init as init


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 调用父类的__init__方法
        conv = [nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
                ]
        self.conv = nn.Sequential(*conv)  # *作用在实参上，代表的是将输入迭代器拆成一个个元素
        fc = [nn.Flatten(),  # 平整化,默认将张量拉成一维的向量
              nn.Linear(in_features=5 * 5 * 16, out_features=120),
              nn.ReLU(inplace=True),
              nn.Linear(in_features=120, out_features=84),
              nn.ReLU(inplace=True),
              nn.Linear(in_features=84, out_features=10),
              ]
        self.fc = nn.Sequential(*fc)



    def _initialize_weights(self,y):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0.1)
                y = 1
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        y = self.conv(x)
        y = self.fc(y)
        return y

if __name__ == '__main__':
    model = CNN()
    # model.train()
    x = torch.randn(1, 1, 32, 32)
    y_out = model.forward(x)
    print("input=", x, '\n', "model=", model, '\n', "output=", y_out)
