import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Create DataFrame
data_path = 'eval/model_results.xlsx'

df = pd.read_excel(data_path)

# Filter for the 'Mean' class and 'train' split
df_filt = df[(df['Class'] == 'Mean')]

# Plot
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))

# Draw the lineplot with confidence intervals
sns.lineplot(
    data=df_filt[(df_filt['Split'] == 'train')],
    x='Epoch',
    y='Loss',
    err_style='band',
    ci='sd',
)
sns.lineplot(
    data=df_filt[(df_filt['Split'] == 'test')],
    x='Epoch',
    y='Dice',
    err_style='band',
    ci='sd',
)

plt.title('F1 Score for Train Subset (Mean Class)')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.show()
print('')
