import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

csv_dir = r"D:\DeepLearning\csv2\potsdam"
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
plt.figure(figsize=(10, 6))

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    epochs = df['epoch'].unique()
    miou_values = []

    for epoch in epochs:
        epoch_data = df[df['epoch'] == epoch]
        miou_value = epoch_data['train_loss'].dropna().values[0]
        miou_values.append(miou_value)

    plt.plot(epochs, miou_values,
             linestyle='-',
             linewidth=1,
             label=os.path.splitext(os.path.basename(csv_file))[0])

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('mIoU(%)', fontsize=14)
plt.title('mIoU vs Epoch', fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(fontsize=12)
plt.tight_layout()
plt.show()