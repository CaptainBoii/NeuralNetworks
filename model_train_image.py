from glob import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

results_labels = ['Test loss', 'Test accuracy', 'F1', 'Balanced', 'Precision', 'Recall']

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

lengths = ['5s', '15s']
genres = ['Classical', 'Electronic', 'Hip-Hop', 'Metal', 'Pop', 'Rock']
img_width, img_height = 250, 250


def read_spectrograms(length_: str) -> (list, list):
    labels = []
    paths = []
    for genre in genres:
        image_files = glob("Spectrograms/" + length_ + '/' + genre + '/*.png')
        for file in image_files:
            paths.append(file)
            labels.append(genre)
    return paths, labels


for length in lengths:
    results = np.zeros((2 * 5, 6))

    X, y = read_spectrograms(length)

    for fold_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train = np.array(X)[train_index]
        y_train = np.array(y)[train_index]
        X_test = np.array(X)[test_index]
        y_test = np.array(y)[test_index]

        # Create an ImageDataGenerator for augmenting and preprocessing the images
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        optimizer = Adam(learning_rate=0.001)  # Adjust learning rate as needed

        # Load and augment the training data
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'X': X_train, 'y': y_train}),
            x_col='X',
            y_col='y',
            target_size=(img_width, img_height),
            batch_size=128,
            class_mode='categorical')

        # Create a CNN model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model_history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // 128,
            epochs=100)

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'X': X_test, 'y': y_test}),
            x_col='X',
            y_col='y',
            target_size=(img_width, img_height),
            batch_size=128,
            class_mode='categorical',
            shuffle=False)

        test_loss, test_acc = model.evaluate(test_generator)
        y_result_pre = model.predict(test_generator)
        y_result = []
        for i, line in enumerate(y_result_pre):
            y_result.append(genres[int(np.array(line).argmax())])

        results[fold_id][0] = test_loss
        results[fold_id][1] = test_acc
        results[fold_id][2] = f1_score(y_test, y_result, average='weighted')
        results[fold_id][3] = balanced_accuracy_score(y_test, y_result)
        results[fold_id][4] = precision_score(y_test, y_result, average='weighted', zero_division='warn')
        results[fold_id][5] = recall_score(y_test, y_result, average='weighted')

    df = pd.DataFrame(results)  # A is a numpy 2d array
    df.to_csv('results_image_' + length + '.csv', header=results_labels, index=False)
