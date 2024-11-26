import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

train_data_directory = "/Users/jival/Documents/Uni Stuff/3rd year/Sem 2/EPE 321/prac/EPE Local Code/Data/train"
test_data_directory = "/Users/jival/Documents/Uni Stuff/3rd year/Sem 2/EPE 321/prac/EPE Local Code/Data/test"
model_file = "/Users/jival/Documents/Uni Stuff/3rd year/Sem 2/EPE 321/prac/EPE Local Code/book_detection_v6.keras"

train_augmentation = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_augmentation = ImageDataGenerator(
    rescale=1.0/255
)

training_data = train_augmentation.flow_from_directory(
    train_data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = validation_augmentation.flow_from_directory(
    test_data_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

pre_trained_model = InceptionV3(input_shape=[224, 224] + [3], weights="imagenet", include_top=False)
# print(model.summary())


for layer in pre_trained_model.layers:
    layer.trainable = False

# Compress model layers using global average pooling
compressed_layer = tf.keras.layers.GlobalAveragePooling2D()(pre_trained_model.output)

# Flatten the compressed layer
temp_layer = Flatten()(compressed_layer)

# Define new output layer
new_out_layer = Dense(2, activation='softmax')(temp_layer)

new_model = Model(inputs=pre_trained_model.input, outputs=new_out_layer)
# print(new_model.summary())

new_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(model_file, verbose=1, save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=30, verbose=1)
]


r = new_model.fit(
    training_data,
    validation_data=test_data,
    epochs=50,
    steps_per_epoch=len(training_data),
    validation_steps=len(test_data),
    callbacks=callbacks
)

