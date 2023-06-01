
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate

# 为每个特征定义输入层
sequence_input = Input(shape=(maxlen,))
physical_input = Input(shape=(maxlen, num_physical_features))
conservation_input = Input(shape=(maxlen,))

# 序列特征 CNN 模型
sequence_cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(sequence_input)
sequence_cnn = MaxPooling1D(pool_size=2)(sequence_cnn)

# 物理化学特征 CNN 模型
physical_cnn = Conv1D(filters=32, kernel_size=3, activation='relu')(physical_input)
physical_cnn = MaxPooling1D(pool_size=2)(physical_cnn)

# 进化保守性特征 dense 模型
conservation_dense = Dense(16, activation='relu')(conservation_input)
conservation_dense = Dense(8, activation='relu')(conservation_dense)

# 将三个模型的输出进行融合
merged = concatenate([sequence_cnn, physical_cnn, conservation_dense])
merged = Flatten()(merged)
merged = Dense(64, activation='relu')(merged)

# 最后的输出
output = Dense(1, activation='sigmoid')(merged)

# 定义模型的输入和输出
model = Model(inputs=[sequence_input, physical_input, conservation_input], outputs=output)
# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 在上面的代码中，我们为每个特征定义了单独的输入层，然后使用单独的CNN模型或Dense模型分别对每个输入层进行处理。最后使用 concatenate 将三个模型的输出连接在一起，然后添加 Flatten 层和Dense层进行联合处理，最后将输出传递到输出层进行二分类。在编译模型时，我们使用二元交叉熵作为损失函数和 Adam 优化器。