import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# dataset - make sure you have this folder structure:
# data/train/class1, data/train/class2 ...
# data/val/class1, data/val/class2 ...

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    "data/train",
    target_size=(224,224),
    batch_size=16
)

val = val_datagen.flow_from_directory(
    "data/val",
    target_size=(224,224),
    batch_size=16
)

# model
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
x = base.output
x = GlobalAveragePooling2D()(x)
out = Dense(train.num_classes, activation="softmax")(x)
model = Model(inputs=base.input, outputs=out)

# freeze base layers (only train top)
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# train
model.fit(train, validation_data=val, epochs=5)

# save
model.save("breed_classifier.h5")

print("✅ Training complete! Model saved as breed_classifier.h5")
