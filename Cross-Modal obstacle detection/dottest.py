from tensorflow.python.keras.layers import Dot, Input, Lambda
from tensorflow.python.keras.models import Model
import numpy as np
import tensorflow.python.keras.backend as K
# 定义一个函数，用于计算两个矩阵的点乘

# 定义一个函数，用于计算两个矩阵的点乘
def matrix_dot(x):
    return K.dot(x[0], x[1])
# 创建两个随机矩阵
a = np.ones((3, 3))
b = np.ones((3, 3))
a=np.asarray([[1,2,3],[1,3, 4],[1, 2,3]],dtype=np.float32)
# 定义输入层
input_a = Input(shape=(3,))
input_b = Input(shape=(3,))
# 创建Lambda层
dot_layer = Lambda(matrix_dot)
# 将输入层传递到Lambda层
output = dot_layer([input_a, input_b])
model = Model(inputs=[input_a, input_b], outputs=output)
print(model.summary())
# 测试模型
output_value = model.predict([a, b])
print(output_value)



