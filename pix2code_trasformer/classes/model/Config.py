__author__ = 'jsc'

CONTEXT_LENGTH = 128
IMAGE_SIZE = 224
BATCH_SIZE = 5
EPOCHS = 50

# device = 'cpu'
device = 'cuda'

# 模型参数
d_lan_encoder_model = 256
d_model = 512  # embedding 维度
d_ff = 2048  # FeedForward dimension 前馈神经网络 提取特征
d_k = d_v = 64  # Q K两个矩阵的维度
n_layers = 6  # encoder和decode的层数
n_heads = 8  # 多头的个数

input_path = "./data/web/all_data"
output_path = "./data/web/out"
# 模型保存位置
PATH = './train_model/trans_pix.pth'