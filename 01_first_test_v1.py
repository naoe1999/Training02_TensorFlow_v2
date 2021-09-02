# 텐서플로우 2.x 설치된 환경에서 1.x 코드를 돌리기 위해 아래와 같이 import
import tensorflow.compat.v1 as tf

# 텐서플로우 2.x에 와서는 eager_execution이 기본적으로 enable 되어 있음
# tf.Session을 사용하기 위해 이를 disable 시켜야 함
tf.disable_eager_execution()

# 여기부터는 기존 코드와 동일
msg = tf.constant('Hello, TensorFlow')
sess = tf.Session()
print(sess.run(msg))

a = tf.constant(1)
b = tf.constant(2)
print(sess.run(a + b))
