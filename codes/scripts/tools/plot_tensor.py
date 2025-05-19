import torch
import matplotlib.pyplot as plt

# 创建一个随机的 PyTorch Tensor，模拟彩色图像数据
# 假设图像大小为 3x100x100，其中 3 表示 RGB 通道
file = "/home/wumenglin/repo-dev/DL-Art-School-dev/codes/speech_embeds2.pt"
tensor = torch.load(file)

# 将 Tensor 转换为 NumPy 数组
# 因为 matplotlib 不直接支持 PyTorch Tensor
numpy_array = tensor.numpy()

# 调整数组的维度顺序，因为 matplotlib 需要 (H, W, C) 的格式
# 而 PyTorch 的 Tensor 默认是 (C, H, W) 的格式
numpy_array = numpy_array.transpose(1, 2, 0)

# 使用 matplotlib 绘制图像
fig, ax = plt.subplots()
cax = ax.imshow(numpy_array)
ax.axis('off')  # 关闭坐标轴


# 添加颜色条
fig.colorbar(cax)

# 保存图像为 PNG 格式
plt.savefig('output_image_with_colorbar.png', bbox_inches='tight', pad_inches=0)

# 显示图像
plt.show()