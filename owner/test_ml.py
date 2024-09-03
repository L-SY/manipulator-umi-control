import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 超参数
batch_size = 64
learning_rate = 0.001
epochs = 2

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# 定义简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # 扁平化图像
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 存储损失
losses = []

# 训练模型
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # 零梯度
        optimizer.zero_grad()
        # 向前传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 个小批量输出一次损失
            avg_loss = running_loss / 100
            losses.append(avg_loss)
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {avg_loss:.4f}')
            running_loss = 0.0

print('Finished Training')

# 可视化训练损失
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Iteration (x100)')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.show()

# 可视化一些预测结果
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 获取一些测试数据
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 显示图像
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{i}' for i in labels))

# 进行预测
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 显示预测结果
print('Predicted: ', ' '.join(f'{i}' for i in predicted))
