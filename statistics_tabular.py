import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

results_initial_5s = pd.read_csv('results_initial_5s.csv')
results_middle_5s = pd.read_csv('results_middle_5s.csv')
results_final_5s = pd.read_csv('results_final_5s.csv')
results_initial_15s = pd.read_csv('results_initial_15s.csv')
results_middle_15s = pd.read_csv('results_middle_15s.csv')
results_final_15s = pd.read_csv('results_final_15s.csv')

results_image_mel_5s = pd.read_csv('results_image_mel_5s.csv')
results_image_spectrogram_5s = pd.read_csv('results_image_spectrogram_5s.csv')
results_image_mel_15s = pd.read_csv('results_image_mel_15s.csv')
results_image_spectrogram_15s = pd.read_csv('results_image_spectrogram_15s.csv')

metrics = ['Test accuracy', 'F1', 'Balanced', 'Precision', 'Recall']

mean_results_initial_5s = results_initial_5s[metrics].mean()
mean_results_middle_5s = results_middle_5s[metrics].mean()
mean_results_final_5s = results_final_5s[metrics].mean()
mean_results_initial_15s = results_initial_15s[metrics].mean()
mean_results_middle_15s = results_middle_15s[metrics].mean()
mean_results_final_15s = results_final_15s[metrics].mean()

mean_results_image_mel_5s = results_image_mel_5s[metrics].mean()
mean_results_image_spectrogram_5s = results_image_spectrogram_5s[metrics].mean()
mean_results_image_mel_15s = results_image_mel_15s[metrics].mean()
mean_results_image_spectrogram_15s = results_image_spectrogram_15s[metrics].mean()

for i in [mean_results_initial_5s, mean_results_initial_15s, mean_results_middle_5s, mean_results_middle_15s,
          mean_results_final_5s, mean_results_final_15s]:
    result = ""
    for j in i:
        result += str(round(j, 3))
        result += " & "
    print(result)

print()
print('======')
print()

t_stat = np.empty(shape=(2, 2))
p_val = np.empty(shape=(2, 2))
results = np.empty(shape=(2, 2), dtype='bool')
alpha = np.empty(shape=(2, 2), dtype='bool')
cross = np.empty(shape=(2, 2), dtype='bool')

res = [results_middle_5s, results_middle_15s]

for i in range(0, 2):
    for j in range(0, 2):
        t, p = stats.ttest_rel(res[i]['Test accuracy'], res[j]['Test accuracy'])
        t_stat[i, j] = t
        p_val[i, j] = p
        results[i, j] = t is not np.nan and t > 0
        alpha[i, j] = t is not np.nan and p < 0.05
        cross[i, j] = alpha[i, j] and results[i, j]

print(t_stat)
print(p_val)
print(results)
print(alpha)
print(cross)

print()
print('======')
print()

for i in [mean_results_image_spectrogram_5s, mean_results_image_spectrogram_15s,
          mean_results_image_mel_5s, mean_results_image_mel_15s]:
    result = ""
    for j in i:
        result += str(round(j, 3))
        result += " & "
    print(result)

t_stat = np.empty(shape=(2, 2))
p_val = np.empty(shape=(2, 2))
results = np.empty(shape=(2, 2), dtype='bool')
alpha = np.empty(shape=(2, 2), dtype='bool')
cross = np.empty(shape=(2, 2), dtype='bool')


res = [results_image_spectrogram_15s, results_image_mel_15s]

for i in range(0, 2):
    for j in range(0, 2):
        t, p = stats.ttest_rel(res[i]['Test accuracy'], res[j]['Test accuracy'])
        t_stat[i, j] = t
        p_val[i, j] = p
        results[i, j] = t is not np.nan and t > 0
        alpha[i, j] = t is not np.nan and p < 0.05
        cross[i, j] = alpha[i, j] and results[i, j]

print(t_stat)
print(p_val)
print(results)
print(alpha)
print(cross)
