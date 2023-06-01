
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, concatenate, Dense, Dropout

# 定义每个特征的输入层
sequence_input = Input(shape=(maxlen,))
physical_input = Input(shape=(maxlen, num_physical_features))
custom_input = Input(shape=(maxlen, num_custom_features))

# 处理序列特征
sequence_model = GRU(units=64, return_sequences=True)(sequence_input)
sequence_model = Dropout(0.1)(sequence_model)

# 处理物理化学特征
physical_model = GRU(units=32, return_sequences=True)(physical_input)
physical_model = Dropout(0.1)(physical_model)

# 处理自定义特征
custom_model = GRU(units=16, return_sequences=True)(custom_input)
custom_model = Dropout(0.1)(custom_model)

# 将三个模型输出拼接在一起
merged_model = concatenate([sequence_model, physical_model, custom_model], axis=2)

# 处理多层循环神经网络
final_model = GRU(units=32)(merged_model)
final_model = Dropout(0.1)(final_model)

# 最后的输出层
output = Dense(1, activation='sigmoid')(final_model)

# 使用多输入层定义模型的输入和输出
model = Model(inputs=[sequence_input, physical_input, custom_input], outputs=output)

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 在上面的代码中，我们定义了3个输入张量，分别用于序列特征，
# 物理化学特征和自定义特征。然后，我们使用单个GRU模型对每个输入进行处理。
# 我们使用将上述三个模型的输出通过 concatenate 层的方式进行拼接，
# 然后再将结果输入到单独的GRU层中进行处理。最后，我们使用单个Dense层进行输出。
# 在编译模型时，使用二元交叉熵作为损失函数和Adam优化器。