import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
import os
import cv2
from matplotlib import pyplot as plt

tf.get_logger().setLevel('ERROR')  # changing the logger level of tensorflow so that it only outputs errors
# setting gpu memory to growable so that I don't get memory error while training
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)


# defining scheduler for variable learning rate during training
def scheduler(epoch, lr):
    if epoch == 10:
        return 0.00001
    elif epoch == 100:
        return 0.0000001
    elif epoch == 200:
        return 0.00000001
    else:
        return lr


# callback for training
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
tf.config.experimental.get_memory_growth = True  # setting memory as growable
path = "C:/Users/Pranshul/Desktop/BB/OUTPUT/vott-csv-export/"  # path to training images and csv file
train_csv = pd.read_csv(os.path.join(path, "BB-export.csv"))  # loading csv file
image_array = []
flip_array = []
# reading images and adding to the array
for img_path in train_csv["image"]:
    img_path = os.path.join(path, img_path)
    img = cv2.imread(img_path)
    image_array.append(img)
    flip_array.append(cv2.flip(img, 1))  # fliping images and adding to the array
image_array = np.array(image_array)
flip_array = np.array(flip_array)
train_y = train_csv.drop(['image', 'label'], axis=1).to_numpy()
flip_y = np.copy(train_y)
# changing the y labels for the flipped images
flip_y[:, 1] = 1824 - flip_y[:, 1]
flip_y[:, 3] = 1824 - flip_y[:, 3]
train_y = np.concatenate([train_y, flip_y], axis=0)
image_array = np.concatenate([image_array, flip_array], axis=0)


# defining function to show images with rectangle drawn
def showimage(image, x1, y1, x2, y2, image_name):
    img = image
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=3)
    cv2.imwrite("C:/Users/Pranshul/Desktop/BB/Testresult/" + image_name, img)


# defining normal cnn model
def Model():
    model = K.models.Sequential()
    model.add(K.layers.Conv2D(128, (3, 3), (2, 2), activation='relu', input_shape=(360, 640, 3),
                              bias_initializer=K.initializers.Ones()))
    # model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Conv2D(256, (3, 3), (2, 2), activation='relu',
                              bias_initializer=K.initializers.Ones()))
    # model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Conv2D(256, (3, 3), (2, 2), activation='relu',
                              bias_initializer=K.initializers.Ones()))
    # model.add(K.layers.Dropout(0.2))
    model.add(K.layers.MaxPool2D())
    model.add(K.layers.Conv2D(64, (3, 1), (1, 2), activation="relu"))
    model.add(K.layers.MaxPool2D())
    # model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Conv2D(4, (10, 10), (2, 2), activation="relu"))
    model.compile(optimizer=K.optimizers.Adam(0.0001), loss=K.losses.mean_squared_logarithmic_error)
    return model


# defining model for one shot
def OneShot():
    INPUT = K.Input(shape=(360, 640, 3))
    model = K.layers.Conv2D(32, (3, 3), (2, 2), activation='relu',
                            bias_initializer=K.initializers.Ones())(INPUT)
    model = K.layers.Conv2D(64, (3, 3), (2, 2), activation='relu', padding="same",
                            bias_initializer=K.initializers.Ones())(model)
    for i in range(5):
        model = K.layers.Conv2D(128, (3, 3), (1, 1), activation='relu', padding="same",
                                bias_initializer=K.initializers.Ones())(model)
        model = K.layers.BatchNormalization()(model)
    model = K.layers.Conv2D(128, (3, 3), (2, 2), activation='relu',
                            bias_initializer=K.initializers.Ones())(model)
    model = K.layers.Conv2D(128, (3, 3), (2, 2), activation="relu", padding='same')(model)
    model = K.layers.MaxPool2D()(model)
    boxes = K.layers.Conv2D(4, (3, 3), (2, 2), activation="relu", trainable=True)(model)
    boxes = K.layers.Reshape((45, 4))(boxes)
    scores = K.layers.Conv2D(1, (3, 3), (2, 2), activation="sigmoid")(model)
    scores = K.layers.Reshape((45, 1))(scores)
    model = K.Model(inputs=INPUT, outputs=[scores, boxes], name="custom_model")
    return model


# Training using normal model
# kmodel = Model()
# kmodel.fit(x=image_array[:80], y=train_y[:80], epochs=epochs, batch_size=2, validation_split=0.1,
#            use_multiprocessing=True)
# Training using OneSot learning with gradient tape.
kmodel = OneShot()
# load weights
kmodel.load_weights("C:/Users/Pranshul/Desktop/BB")
kmodel.save("C:/Users/Pranshul/PycharmProjects/BBProject/My_Model")
epochs = 470
optimizer = K.optimizers.Adam(learning_rate=5e-7)
loss_fn = K.losses.MeanSquaredLogarithmicError()
for epoch in range(epochs):
    for i in range(0, image_array.shape[0]):
        with tf.GradientTape() as tape:
            score, boxes = kmodel(image_array[i].reshape(1, 360, 640, 3), training=True)
            boxes = boxes[0]
            score = score[0]
            selected_indices = tf.image.non_max_suppression(
                boxes, tf.squeeze(score), 1, iou_threshold=0.7,
                score_threshold=float('-inf'),
            )
            box = tf.gather(boxes, selected_indices)
            loss = loss_fn(train_y[i], box)
            loss_val = K.backend.get_value(loss)

        grads = tape.gradient(loss, kmodel.trainable_weights)
        optimizer.apply_gradients(zip(grads, kmodel.trainable_weights))
    print("epoch: " + str(epoch) + "\n" + " loss: " + str(loss_val))
kmodel.save_weights("C:/Users/Pranshul/Desktop/BB")  # saving weights
# reading video from local file for testing
cap = cv2.VideoCapture("C:/Users/Pranshul/Desktop/BB/videoplayback.mp4")
cap.set(cv2.CAP_PROP_FPS, 60)
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        score, boxes = kmodel(frame.reshape(1, 360, 640, 3), training=False)
        boxes = boxes[0]
        score = score[0]
        selected_indices = tf.image.non_max_suppression(
            boxes, tf.squeeze(score), 1, iou_threshold=0.7,
            score_threshold=float('-inf'),
        )
        points = tf.gather(boxes, selected_indices)
        points = points.numpy()
        cv2.rectangle(frame, pt1=(points[0][0], points[0][1]), pt2=(points[0][2], points[0][3]),
                      color=(0, 0, 255), thickness=3)
        cv2.imshow("win", frame)
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
