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
    config_name='draw_boxplots',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Define absolute paths
    metrics_path = os.path.join(PROJECT_DIR, cfg.metrics_path)
    save_dir = os.path.join(PROJECT_DIR, cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Read DataFrame with metrics
    df = pd.read_csv(metrics_path)

    # Plotting
    sns.set(style='whitegrid')

    # Iterate over columns except 'Model', 'Fold', and 'Epoch' to create boxplots
    classes = df.columns.difference(['Model', 'Fold', 'Epoch'])
    for class_name in classes:
        plt.figure(figsize=(12, 12))  # Adjust figure size for better presentation

        # Create a new palette for each figure
        palette = sns.color_palette('bright', 6)

        ax = sns.boxplot(
            x='Model',
            y=class_name,
            data=df,
            palette=palette,
            showfliers=False,
        )
        plt.ylabel('DSC', fontsize=36)
        plt.xticks(rotation=90, fontsize=30)
        plt.yticks(fontsize=30)
        ax.set_xlabel('')
        ax.set_ylim(0, 1)
        sns.despine()
        plt.tight_layout()

        # Save the plot as a high-quality image file in PNG format
        save_path = os.path.join(save_dir, f'{class_name}_boxplot.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()  # Close the current plot to release memory


if __name__ == '__main__':
    main()
