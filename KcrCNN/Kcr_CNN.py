
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 构建模型
model = Sequential()

# 添加卷积层1和池化层1
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(maxlen, embed_dim)))
model.add(MaxPooling1D(pool_size=2))

# 添加卷积层2和池化层2
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# 添加卷积层3和池化层3
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# 添加平展层和全连接层
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#
# 在上面的代码中，我们使用了三个卷积层和池化层。第一层卷积层的卷积核有 64 个，大小为 3，激活函数为 ReLU。第一层池化层的池化窗口大小为 2。第二层和第三层的设置类似，不同的是滤波器数量和大小不同。最后，使用平展层和全连接层实现分类，输出 0 或 1。
#
# 我们使用交叉熵作为损失函数，使用 Adam 优化器来优化模型。