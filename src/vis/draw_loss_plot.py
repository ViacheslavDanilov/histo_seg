import logging
import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

from src import PROJECT_DIR

log = logging.getLogger(__name__)


@hydra.main(
    config_path=os.path.join(PROJECT_DIR, 'configs'),
    config_name='draw_loss_plot',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    metrics_path = os.path.join(PROJECT_DIR, cfg.metrics_path)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Read DataFrame with metrics
    df = pd.read_excel(metrics_path)
    gb = df.groupby('Model')

    for model_name, df_model in gb:
        # Filter for the 'Mean' class
        df_filt = df_model[df_model['Class'] == 'Mean']

        # Plot
        sns.set(style='whitegrid')
        plt.figure(figsize=(12, 8))

        # Customize color palette
        palette = sns.color_palette('bright', 2)

        # Draw line plots with confidence intervals
        sns.lineplot(
            data=df_filt[df_filt['Split'] == 'train'],
            x='Epoch',
            y='Loss',
            color=palette[0],
            label='Loss (Train)',
            err_style='band',
            errorbar='sd',
        )
        sns.lineplot(
            data=df_filt[df_filt['Split'] == 'test'],
            x='Epoch',
            y='Dice',
            color=palette[1],
            label='DSC (Test)',
            err_style='band',
            errorbar='sd',
        )

        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Metric Value', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)

        # Set coordinate axis limits
        plt.ylim(0, 1)
        plt.xlim(0, 125)
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(save_dir, f'{model_name}_loss.png')
        plt.savefig(save_path, dpi=600)
        plt.show()


if __name__ == '__main__':
    main()
