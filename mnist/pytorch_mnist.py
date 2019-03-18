import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os

# Hyper Parameters
EPOCH = 1
LR = 0.001
BATCHSIZE = 50
DOWNLOAD_MNIST = False

# prepare the dataset

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    download=DOWNLOAD_MNIST,
)

# print train example
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title("%i" % train_data.train_labels[0])
# plt.show()

# load data
train_loder = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCHSIZE,
    shuffle=True,
    # num_workers=2,
)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.    # shape (2000, 1, 28, 28)
test_y = test_data.test_labels[:2000]                                                       # value in range(0,1)

# bulid a net
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(   # this is a simple method
            nn.Conv2d(                # shape(1,28,28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),                    # shape(16,28,28)
            nn.MaxPool2d(kernel_size=2),  # shape(16,14,14)
        )
        self.conv2 = nn.Sequential(         # shape(32,7,7)
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out, x


net = ConvNet()
# print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# # visualization
# from matplotlib import cm
# try:
#     from sklearn.manifold import TSNE
#     HAS_SK = True
# except:
#     HAS_SK = False
#     print('Please install sklearn for layer visualization')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9))
#         plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max())
#     plt.ylim(Y.min(), Y.max())
#     plt.title('Visualize last layer')
#     plt.show()
#     plt.pause(0.01)

# plt.ion()

# begin to training
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loder):
        output = net(b_x)[0]
        # print(output)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output, last_layer = net(test_x)
            temp = torch.max(test_output, 1)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#             if HAS_SK:
#                 # Visualization of trained flatten layer (T-SNE)
#                 tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#                 plot_only = 500
#                 low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
#                 labels = test_y.numpy()[:plot_only]
#                 plot_with_labels(low_dim_embs, labels)
#
# plt.ioff()

# print 10 predictions from test data
test_output, _ = net(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

