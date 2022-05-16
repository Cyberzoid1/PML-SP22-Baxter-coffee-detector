
# https://www.youtube.com/watch?v=dFdMyUbtKM4

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import keras_tuner
import matplotlib.pyplot as plt

image_size = (102, 85)
batch_size = 64
random_seed = 34855

# https://keras.io/api/preprocessing/image/
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "images-sm",
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="training",
    seed=random_seed,
    color_mode="grayscale",
    image_size=image_size,
    shuffle=True,
    interpolation="bilinear", 
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "images-sm",
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",
    seed=random_seed,
    color_mode="grayscale",
    image_size=image_size,
    shuffle=True,
    interpolation="bilinear", 
    batch_size=batch_size
)

# # Visualize the data
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images(i).numpy().astype("uint8"))
#         plt.title(int(lables[i]))
#         plt.axis("off")


# Data augmentation
# https://keras.io/guides/preprocessing_layers/
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomRotation(0.005, seed=random_seed), # Ad rotation noise
        layers.experimental.preprocessing.RandomContrast(0.05, seed=random_seed)   # Contrast
    ]
)

print("\nAugmenting data")
# Visualize augmentation
# plt.figure(figsize=(12, 12))
# for images, _ in train_ds.take(5):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")
#     plt.show()
# exit()


# CPU data augmentation
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))

# GPU data augmentation
# inputs = keras.Input(shape=input_shape)
# x = data_augmentation(inputs)
# x = layers.experimental.preprocessing.Rescaling(1./255)(x)


# Configure dataset for performance
train_ds = train_ds.prefetch(buffer_size=32)
augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def make_model():
    input_shape = image_size + (1,)
    num_classes = 6
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0/128)(inputs)
    x = layers.Conv2D(32, 3,input_shape=(image_size, 1), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(20, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

def make_tuned_model(hp):
    input_shape = image_size + (1,)
    num_classes = 6
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0/255)(inputs) #(x)
    x = layers.Conv2D(32, 3,input_shape=(image_size, 1), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Number of layers of the CNN is also a hyperparameter.
    for i in range(hp.Int("cnn_layers", 1, 3)):
        x = layers.Conv2D(
            hp.Int(f"filters_{i}", 32, 128, step=32),
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(20, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


# Create model
input_shape = image_size + (1,)
print("Input shape: %s" % str(input_shape))
model = make_model()
print(model.summary())
keras.utils.plot_model(model, show_shapes=True)


# # Tune
# print("\nTuning")
# hp = keras_tuner.HyperParameters()
# hp.values["model_type"] = "cnn"
# model = make_tuned_model(hp)
# model.compile(
#     optimizer=optimizers.Adam(1e-3),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )

# tuner = keras_tuner.RandomSearch(
#     model,
#     max_trials=10,
#     # Do not resume the previous search in the same directory.
#     overwrite=True,
#     seed=1,
#     objective="val_accuracy",
#     # Set a directory to store the intermediate results.
#     directory="tuner",
# )

# tuner.search(
#     augmented_train_ds,
#     validation_split=0.2,
#     epochs=2
# )
# tuner.results_summary()
# exit()

# Train
epochs = 2000
callbacks = [
    #keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, min_delta=0.01),
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    keras.callbacks.CSVLogger("epoch-history.csv", separator=",", append=True)
]

print("\nCompiling")
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nTraining")
history = model.fit(
    augmented_train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds
)
model.save('trained_model.savemodel')


print("\nEvaluating")
# Evaluate on training data
_, train_acc = model.evaluate(train_ds, verbose=1)
# Evaluate on test data
_, test_acc = model.evaluate(val_ds, verbose=1)

print('training acc is', train_acc)
print('test acc is', test_acc)


# Plot training results
# Source https://stackoverflow.com/a/56807595
plot1 = plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# plt.show()

plot2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.show()
