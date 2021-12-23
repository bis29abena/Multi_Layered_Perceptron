# Usage
# python main.py --output Output\output.png --save Output\image.png --verify Output\verify.png

# import the necessary packages
from bis29_.plotting.plot import Plot
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct an argument parser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to plots")
ap.add_argument("-s", "--save", required=True, help="path to save image")
ap.add_argument("-v", "--verify", required=True, help="path to save verified prediction")
args = vars(ap.parse_args())

# grab the mnis datasets
print("[INFO] accessing the mnist....")
((trainX, trainY), (testX, testY)) = mnist.load_data()

# scale the data to a pixel range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Instantiate the plot class and display the first 25 images in the training dataset
fig = Plot()
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    fig.show_images(i, images=trainX, labels=trainY)
plt.savefig(args["save"])
plt.show()

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the architecture using keras
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(256, activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network.....")
sgd = SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

# evaluate the network
print("[INFO] evaluating the network....")
target_names = [str(x) for x in lb.classes_]
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=target_names))

# plotting the predictions
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
fig.plot_image(i, predictions_array=predictions[i], true_label=testY, img=testX, class_names=target_names)
plt.subplot(1, 2, 2)
fig.plot_value_array(i, predictions_array=predictions[i], true_label=testY)
plt.savefig(args["verify"])


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.ylabel("Loss\Accuracy")
plt.xlabel("Epoch #")
plt.legend()
plt.savefig(args["output"])
plt.show()