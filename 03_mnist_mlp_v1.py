import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.datasets import mnist

tf.disable_eager_execution()

# 기존의 데이터 로딩 방식. 지금은 작동 안함
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# keras를 사용한 새로운 방식
# MNIST 데이터를 다운로드 합니다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# 28*28 형태의 이미지를 784차원으로 flattening 합니다.
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.

# 레이블 데이터에 one-hot encoding을 적용합니다.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# 학습을 위한 설정값들을 정의합니다.
learning_rate = 0.001
num_epochs = 30     # 학습횟수
batch_size = 64     # 배치개수
display_step = 1    # 손실함수 출력 주기
input_size = 784    # 28 * 28
hidden1_size = 256
hidden2_size = 64
output_size = 10

# keras mnist를 사용하여 train data 구성
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(batch_size)

iterator = train_data.make_initializable_iterator()
next_batch = iterator.get_next()


# 입력값과 출력값을 받기 위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, output_size])


# ANN 모델을 정의합니다.
def build_ANN(x):
    W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    W_output = tf.Variable(tf.random_normal(shape=[hidden2_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape=[output_size]))

    H1_output = tf.nn.relu(tf.matmul(x, W1) + b1)
    H2_output = tf.nn.relu(tf.matmul(H1_output, W2) + b2)
    logits = tf.matmul(H2_output,W_output) + b_output

    return logits


# ANN 모델을 선언합니다.
predicted_value = build_ANN(x)

# 손실함수와 옵티마이저를 정의합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_value, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 세션을 열고 그래프를 실행합니다.
with tf.Session() as sess:
    # 변수들에 초기값을 할당합니다.
    sess.run(tf.global_variables_initializer())

    sess.run(iterator.initializer)

    # 지정된 횟수만큼 최적화를 수행합니다.
    for epoch in range(num_epochs):
        average_loss = 0.

        # 전체 배치를 불러옵니다.
        total_batch = int(x_train.shape[0] / batch_size)

        # 모든 배치들에 대해서 최적화를 수행합니다.
        for i in range(total_batch):
            batch_x, batch_y = sess.run(next_batch)
            _, current_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})

            average_loss += current_loss / total_batch

        # 지정된 epoch마다 학습결과를 출력합니다.
        if epoch % display_step == 0:
            print("반복(Epoch): %d, 손실 함수(Loss): %f" % ((epoch+1), average_loss))

    # 테스트 데이터를 이용해서 학습된 모델이 얼마나 정확한지 정확도를 출력합니다.
    correct_prediction = tf.equal(tf.argmax(predicted_value, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("정확도(Accuracy): %f" % (accuracy.eval(feed_dict={x: x_test, y: y_test})))
