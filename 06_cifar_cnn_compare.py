import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt


# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.
# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(y_train_one_hot.shape, y_test_one_hot.shape)


# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
train_data = train_data.repeat().shuffle(50000).batch(128)
train_data_iter = iter(train_data)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
test_data = test_data.batch(1000)
test_data_iter = iter(test_data)


# CNN 모델을 정의합니다.
class CNN_VGG16(Model):
    # CNN 모델을 위한 tf.Variable들을 정의합니다.
    def __init__(self, pretrained, trainable):
        vgg = VGG16(weights='imagenet' if pretrained else None, include_top=False, input_shape=(32, 32, 3))
        vgg.trainable = trainable

        x = vgg.get_layer('block3_conv3').output
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = tf.keras.layers.Dropout(0.0, name='dropout')(x)
        logits = tf.keras.layers.Dense(10, name='prediction')(x)
        super(CNN_VGG16, self).__init__(inputs=vgg.input, outputs=logits)

    def __call__(self, x, rate):
        self.get_layer('dropout').rate = rate
        logits = super(CNN_VGG16, self).__call__(x)
        y_pred = tf.nn.softmax(logits)

        return y_pred, logits

    def set_cnn_trainable(self, trainable):
        self.trainable = trainable


# cross-entropy 손실 함수를 정의합니다.
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


# 모델의 정확도를 출력하는 함수를 정의합니다.
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


# 최적화를 위한 RMSprop 옵티마이저를 정의합니다.
optimizer = tf.optimizers.RMSprop(1e-3)


# 최적화를 위한 function을 정의합니다.
def train_step_wrap():
  @tf.function
  def train_step(model, x, y, rate):
      with tf.GradientTape() as tape:
          y_pred, logits = model(x, rate)
          loss = cross_entropy_loss(logits, y)
      gradients = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return train_step


# 1. Random initialized model
CNN_random_model = CNN_VGG16(pretrained=False, trainable=True)

train_random = train_step_wrap()

iter_rd = []
loss_rd = []
acc_rd = []

# 10000 Step만큼 최적화를 수행합니다.
for i in range(10000):
    batch_x, batch_y = next(train_data_iter)

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 0:
        train_accuracy = compute_accuracy(CNN_random_model(batch_x, 0.0)[0], batch_y)
        train_loss = cross_entropy_loss(CNN_random_model(batch_x, 0.0)[1], batch_y)

        print("반복(Iteration): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, train_loss))

        iter_rd.append(i)
        loss_rd.append(train_loss)
        acc_rd.append(train_accuracy)

    # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
    train_random(CNN_random_model, batch_x, batch_y, 0.2)


# 2. Pre-trained model
CNN_pretrained_model = CNN_VGG16(pretrained=True, trainable=True)

train_pretrained = train_step_wrap()

iter_pt = []
loss_pt = []
acc_pt = []

# 10000 Step만큼 최적화를 수행합니다.
for i in range(10000):
    batch_x, batch_y = next(train_data_iter)

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 0:
        train_accuracy = compute_accuracy(CNN_pretrained_model(batch_x, 0.0)[0], batch_y)
        train_loss = cross_entropy_loss(CNN_pretrained_model(batch_x, 0.0)[1], batch_y)

        print("반복(Iteration): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, train_loss))

        iter_pt.append(i)
        loss_pt.append(train_loss)
        acc_pt.append(train_accuracy)

    # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
    train_pretrained(CNN_pretrained_model, batch_x, batch_y, 0.2)


# 3. Pre-trained model with manipulated training schedule
CNN_manipulated_model = CNN_VGG16(pretrained=True, trainable=False)
CNN_manipulated_model.summary()

# 최적화를 위한 RMSprop 옵티마이저를 재정의합니다.
optimizer = tf.optimizers.RMSprop(1e-3)
train_firststep = train_step_wrap()

iter_ma = []
loss_ma = []
acc_ma = []

# Weight 동결 후 1000 Step만큼 최적화를 수행
for i in range(1000):
    batch_x, batch_y = next(train_data_iter)

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 0:
        train_accuracy = compute_accuracy(CNN_manipulated_model(batch_x, 0.0)[0], batch_y)
        train_loss = cross_entropy_loss(CNN_manipulated_model(batch_x, 0.0)[1], batch_y)

        print("반복(Iteration): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, train_loss))

        iter_ma.append(i)
        loss_ma.append(train_loss)
        acc_ma.append(train_accuracy)

    # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
    train_firststep(CNN_manipulated_model, batch_x, batch_y, 0.2)

# Weight 동결 해제 후 나머지 Step만큼 최적화를 수행
CNN_manipulated_model.set_cnn_trainable(True)
CNN_manipulated_model.summary()

optimizer = tf.optimizers.RMSprop(5e-4)
train_secondstep = train_step_wrap()

for i in range(1000, 10000):
    batch_x, batch_y = next(train_data_iter)

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 0:
        train_accuracy = compute_accuracy(CNN_manipulated_model(batch_x, 0.0)[0], batch_y)
        train_loss = cross_entropy_loss(CNN_manipulated_model(batch_x, 0.0)[1], batch_y)

        print("반복(Iteration): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, train_loss))

        iter_ma.append(i)
        loss_ma.append(train_loss)
        acc_ma.append(train_accuracy)

    # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
    train_secondstep(CNN_manipulated_model, batch_x, batch_y, 0.2)


plt.plot(iter_rd, loss_rd, 'b:', iter_rd, acc_rd, 'b-')
plt.plot(iter_pt, loss_pt, 'r:', iter_pt, acc_pt, 'r-')
plt.plot(iter_ma, loss_ma, 'g:', iter_ma, acc_ma, 'g-')
plt.ylim(0, 2)
plt.show()


# 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
test_accuracy_rd = 0.0
test_accuracy_pt = 0.0
test_accuracy_ma = 0.0

for i in range(10):
    test_batch_x, test_batch_y = next(test_data_iter)
    test_accuracy_rd = test_accuracy_rd + compute_accuracy(CNN_random_model(test_batch_x, 0.0)[0], test_batch_y).numpy()
    test_accuracy_pt = test_accuracy_pt + compute_accuracy(CNN_pretrained_model(test_batch_x, 0.0)[0], test_batch_y).numpy()
    test_accuracy_ma = test_accuracy_ma + compute_accuracy(CNN_manipulated_model(test_batch_x, 0.0)[0], test_batch_y).numpy()

test_accuracy_rd = test_accuracy_rd / 10
test_accuracy_pt = test_accuracy_pt / 10
test_accuracy_ma = test_accuracy_ma / 10

print("(random) 테스트 데이터 정확도: %f" % test_accuracy_rd)
print("(pretrained) 테스트 데이터 정확도: %f" % test_accuracy_pt)
print("(manipulated) 테스트 데이터 정확도: %f" % test_accuracy_ma)
