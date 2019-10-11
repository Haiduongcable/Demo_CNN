from keras import models 
from keras import layers 
#Xây dựng lớp convolution
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu' , input_shape = (28,28,1)))
# Xây dựng lớp Maxpooling
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
#Flatten để đưa vào mạng fully connted
model.add(layers.Flatten())

model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10,activation = 'softmax'))
# load dữ liệu
from keras.datasets import mnist
from keras.utils import to_categorical
(train_data,train_label) , (test_data,test_label) = mnist.load_data()

train_data = train_data.reshape((60000,28,28,1))
# normalization dữ liệu bằng scaling
train_data = train_data.astype('float32') / 255
test_data = test_data.reshape((10000,28,28,1))
test_data = test_data.astype('float32') / 255
# Dùng one_hot_encoding
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
# train model với phương pháp tối ưu rmsprop, loss_function : cross entropy, phương thức đánh giá: accuracy cho bài toán classification

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#train model với batch_size 64, số epochs = 5 ( đã được hiệu chỉnh tránh overfitting)
model.fit(train_data,train_label,epochs = 5, batch_size= 64)
# đánh giá model 
test_loss, test_acc = model.evaluate(test_data, test_label)



