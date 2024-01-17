import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


bar_width = 0.25
index = np.arange(len(metrics))

plt.bar(index - bar_width, mean_results_initial_5s, bar_width, label='5s')
plt.bar(index, mean_results_initial_15s, bar_width, label='15s')
plt.xlabel('Metrics')
plt.ylabel('Mean Values')
plt.title('Comparison of Initial model for different lengths')
plt.xticks(index, metrics)  # Set X-axis ticks
plt.legend()
plt.show()
plt.savefig('initial.png')
plt.clf()

plt.bar(index - bar_width, mean_results_middle_5s, bar_width, label='5s')
plt.bar(index, mean_results_middle_15s, bar_width, label='15s')
plt.xlabel('Metrics')
plt.ylabel('Mean Values')
plt.title('Comparison of Middle model for different lengths')
plt.xticks(index, metrics)  # Set X-axis ticks
plt.legend()
plt.show()
plt.savefig('middle.png')
plt.clf()

plt.bar(index - bar_width, mean_results_final_5s, bar_width, label='5s')
plt.bar(index, mean_results_final_15s, bar_width, label='15s')
plt.xlabel('Metrics')
plt.ylabel('Mean Values')
plt.title('Comparison of Last model for different lengths')
plt.xticks(index, metrics)  # Set X-axis ticks
plt.legend()
plt.show()
plt.savefig('final.png')
plt.clf()

plt.bar(index - bar_width, mean_results_initial_5s, bar_width, label='Initial')
plt.bar(index, mean_results_middle_5s, bar_width, label='Middle')
plt.bar(index + bar_width, mean_results_final_5s, bar_width, label='Last')
plt.xlabel('Metrics')
plt.ylabel('Mean Values')
plt.title('Comparison of different models for 5s')
plt.xticks(index, metrics)  # Set X-axis ticks
plt.legend()
plt.show()
plt.savefig('5s_models.png')
plt.clf()

plt.bar(index - bar_width, mean_results_initial_15s, bar_width, label='Initial')
plt.bar(index, mean_results_middle_15s, bar_width, label='Middle')
plt.bar(index + bar_width, mean_results_final_15s, bar_width, label='Last')
plt.xlabel('Metrics')
plt.ylabel('Mean Values')
plt.title('Comparison of different models for 15s')
plt.xticks(index, metrics)  # Set X-axis ticks
plt.legend()
plt.show()
plt.savefig('15s_models.png')
plt.clf()

plt.bar(index - bar_width, mean_results_image_mel_5s, bar_width, label='MEL')
plt.bar(index, mean_results_image_spectrogram_5s, bar_width, label='Spectrogram')
plt.xlabel('Metrics')
plt.ylabel('Mean Values')
plt.title('Comparison of spectrogram types for 5s')
plt.xticks(index, metrics)  # Set X-axis ticks
plt.legend()
plt.show()
plt.savefig('spect_5s.png')
plt.clf()

plt.bar(index - bar_width, mean_results_image_mel_15s, bar_width, label='MEL')
plt.bar(index, mean_results_image_spectrogram_15s, bar_width, label='Spectrogram')
plt.xlabel('Metrics')
plt.ylabel('Mean Values')
plt.title('Comparison of spectrogram types for 15s')
plt.xticks(index, metrics)  # Set X-axis ticks
plt.legend()
plt.show()
plt.savefig('spect_15s.png')

plt.clf()



