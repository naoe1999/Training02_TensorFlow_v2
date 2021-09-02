import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train / 255., x_test / 255.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

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
    def __init__(self):
        cnn = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
        cnn.trainable = True
        super(CNN_VGG16, self).__init__(inputs=cnn.input, outputs=cnn.get_layer('block3_conv3').output)

        self.gap = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
        self.dp = tf.keras.layers.Dropout
        self.fc = tf.keras.layers.Dense(10, name='prediction')

    def __call__(self, x, rate=0.0):
        x = super(CNN_VGG16, self).__call__(x)
        x = self.gap(x)
        x = self.dp(rate)(x)
        logits = self.fc(x)
        y_pred = tf.nn.softmax(logits)

        return y_pred, logits

    def get_featuremap(self, x):
        return super(CNN_VGG16, self).__call__(x)


# cross-entropy 손실 함수를 정의합니다.
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


# 최적화를 위한 RMSprop 옵티마이저를 정의합니다.
optimizer = tf.optimizers.RMSprop(1e-3)


# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(model, x, y, rate):
    with tf.GradientTape() as tape:
        y_pred, logits = model(x, rate)
        loss = cross_entropy_loss(logits, y)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# 모델의 정확도를 출력하는 함수를 정의합니다.
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


# Convolutional Neural Networks(CNN) 모델을 선언합니다.
CNN_model = CNN_VGG16()

# 10000 Step만큼 최적화를 수행합니다.
for i in range(10000):
    batch_x, batch_y = next(train_data_iter)

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 0:
        train_accuracy = compute_accuracy(CNN_model(batch_x, 0.0)[0], batch_y)
        loss_print = cross_entropy_loss(CNN_model(batch_x, 0.0)[1], batch_y)

        print("반복(Iteration): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))

    if i % 1000 == 0:
        CNN_model.save('cifar_partial_vgg16.h5')

    # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
    train_step(CNN_model, batch_x, batch_y, 0.2)

# 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
test_accuracy = 0.0
for i in range(10):
    test_batch_x, test_batch_y = next(test_data_iter)
    test_accuracy = test_accuracy + compute_accuracy(CNN_model(test_batch_x, 0.0)[0], test_batch_y).numpy()
test_accuracy = test_accuracy / 10
print("테스트 데이터 정확도: %f" % test_accuracy)

# 모델 저장
CNN_model.save('cifar_partial_vgg16_final.h5')
