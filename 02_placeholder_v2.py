import tensorflow as tf
import numpy as np


# 텐서플로 2.x에서는 placeholder를 사용하지 않음
# function을 정의하고 인자로 전달
@tf.function
def adder_func(x, y):
    return x + y


print(adder_func(3, 4.5).numpy())
print(adder_func(np.array([1, 3]), np.array([2, 4])).numpy())


@tf.function
def add_and_triple(x, y):
    return 3 * adder_func(x, y)


print(add_and_triple(3, 4.5).numpy())
