import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import RepeatedStratifiedKFold

fitter = StandardScaler()
convertor = LabelEncoder()

results_labels = ['Test loss', 'Test accuracy', 'F1', 'Balanced', 'Precision', 'Recall']

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

lengths = ['5s', '15s']

for length in lengths:
    results = np.zeros((2 * 5, 6))
    datafile = pd.read_csv(length + '_normals.csv')
    datafile = datafile.drop('fourier_tempogram', axis=1)
    class_list = datafile.iloc[:, 0]

    y = convertor.fit_transform(class_list)
    X = fitter.fit_transform(np.array(datafile.iloc[:, 1:], dtype=float))

    for fold_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        model = Sequential([
            Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.7),
            Dense(6, activation='softmax'),
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics='accuracy')
        model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000, batch_size=128)

        test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
        y_result_pre = model.predict(X_test)
        y_result = np.zeros((len(y_test)))
        for i, line in enumerate(y_result_pre):
            y_result[i] = int(np.array(line).argmax())

        results[fold_id][0] = test_loss
        results[fold_id][1] = test_acc
        results[fold_id][2] = f1_score(y_test, y_result, average='weighted')
        results[fold_id][3] = balanced_accuracy_score(y_test, y_result)
        results[fold_id][4] = precision_score(y_test, y_result, average='weighted')
        results[fold_id][5] = recall_score(y_test, y_result, average='weighted')

    df = pd.DataFrame(results)  # A is a numpy 2d array
    df.to_csv('results_' + length + '.csv', header=results_labels, index=False)
