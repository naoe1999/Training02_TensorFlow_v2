import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt


train_mode = False


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


# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
train_data = train_data.repeat().shuffle(50000).batch(128)
train_data_iter = iter(train_data)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
test_data = test_data.batch(1000)
test_data_iter = iter(test_data)


label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# CNN 모델을 정의합니다.
class CNN_VGG16(Model):
    # CNN 모델을 위한 tf.Variable들을 정의합니다.
    def __init__(self, pretrained=True, trainable=True):
        vgg = VGG16(weights='imagenet' if pretrained else None, include_top=False, input_shape=(32, 32, 3))
        vgg.trainable = trainable

        x = vgg.get_layer('block3_conv3').output
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = tf.keras.layers.Dropout(0.0, name='dropout')(x)
        logits = tf.keras.layers.Dense(10, name='prediction')(x)
        super(CNN_VGG16, self).__init__(inputs=vgg.input, outputs=logits)

    def __call__(self, x, rate=0.0):
        self.get_layer('dropout').rate = rate
        logits = super(CNN_VGG16, self).__call__(x)
        y_pred = tf.nn.softmax(logits)
        return y_pred, logits

    def get_featuremap(self, x):
        intermediate_model = Model(inputs=self.inputs, outputs=self.get_layer('block3_conv3').output)
        return intermediate_model(x)


# cross-entropy 손실 함수를 정의합니다.
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


# 최적화를 위한 RMSprop 옵티마이저를 정의합니다.
optimizer = tf.optimizers.RMSprop(1e-3)


# 모델의 정확도를 출력하는 함수를 정의합니다.
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(model, x, y, rate):
    with tf.GradientTape() as tape:
        _, logits = model(x, rate)
        loss = cross_entropy_loss(logits, y)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if train_mode:
    # Convolutional Neural Networks(CNN) 모델을 선언합니다.
    model = CNN_VGG16(pretrained=True, trainable=True)

    # 10000 Step만큼 최적화를 수행합니다.
    for i in range(10000):
        batch_x, batch_y = next(train_data_iter)

        # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
        if i % 100 == 0:
            train_accuracy = compute_accuracy(model(batch_x, 0.0)[0], batch_y)
            train_loss = cross_entropy_loss(model(batch_x, 0.0)[1], batch_y)

            print("반복(Iteration): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, train_loss))

        # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
        train_step(model, batch_x, batch_y, 0.2)

    print('model training finished.')

    model.save('cifar_vgg16.h5')
    print('model saved.')

else:
    # model = tf.keras.models.load_model('cifar_vgg16.h5', custom_objects={'CNN_VGG16':CNN_VGG16})
    model = CNN_VGG16(pretrained=True, trainable=True)
    model.load_weights('cifar_vgg16.h5')

    print('model restored.')


# 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
test_accuracy = 0.0

for i in range(10):
    test_batch_x, test_batch_y = next(test_data_iter)
    test_accuracy = test_accuracy + compute_accuracy(model(test_batch_x, 0.0)[0], test_batch_y).numpy()

test_accuracy = test_accuracy / 10

print("테스트 데이터 정확도: %f" % test_accuracy)


# sample 이미지 테스트
img = x_test[-2]
label = int(y_test[-2])
print('ground truth:', label, label_names[label])

print(img.shape)
plt.imshow(img)
plt.show()


img_feed = img.reshape((1, 32, 32, 3))

pred = model(img_feed, 0.0)[0]
print(type(pred))

pred = pred.numpy()
cls = np.argmax(pred)

print('prediction:', cls, label_names[cls])


# feature map
features = model.get_featuremap(img_feed)
print(type(features))
print(features.shape)

features = tf.image.resize(features, (32, 32))
print(type(features))
print(features.shape)

features = features.numpy()

# sample channel
plt.imshow(features[0, :, :, 38])
plt.show()

plt.imshow(features[0, :, :, 130])
plt.show()


# weight
weights = model.get_layer('prediction').get_weights()
print(type(weights[0]))
print(weights[0].shape)
print(weights[1].shape)

wcls = weights[0][:, cls]
print(wcls.shape)

i = np.argmax(wcls)
j = np.argmin(wcls)

print(i, wcls[i], j, wcls[j])


feat = features.reshape((32*32, 256))
print(feat.shape)

# class activation map
cam = np.dot(feat, wcls)
cam = cam - np.min(cam)
cam = cam / np.max(cam)

cam = cam.reshape((32, 32))

plt.imshow(cam, cmap=plt.cm.jet, interpolation='nearest', vmin=0, vmax=1)
plt.colorbar(label='CAM')
plt.show()


plt.imshow(img)
plt.show()

plt.imshow(img)
plt.imshow(cam, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
plt.colorbar(label='CAM')
plt.show()

