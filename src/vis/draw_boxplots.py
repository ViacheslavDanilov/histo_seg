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

    # Define y-limits for each class
    y_limits = {
        'Arteriole lumen': (0.5, 1),
        'Arteriole media': (0.5, 1),
        'Arteriole adventitia': (0.5, 1),
        'Venule lumen': (0.5, 1),
        'Venule wall': (0.5, 1),
        'Capillary lumen': (0.5, 1),
        'Capillary wall': (0.5, 1),
        'Immune cells': (0.5, 1),
        'Nerve trunks': (0.5, 1),
        'Mean': (0.5, 1),
    }

    # Define the order of x-axis categories
    model_order = [
        'U-Net',
        'LinkNet',
        'FPN',
        'PSPNet',
        'DeepLabV3',
        'MA-Net',
    ]

    # Plotting
    sns.set(style='whitegrid')

    # Iterate over columns except 'Model', 'Fold', and 'Epoch' to create boxplots
    classes = df.columns.difference(['Model', 'Fold', 'Epoch'])
    for class_name in classes:
        plt.figure(figsize=(12, 12))

        # Create a new palette for each figure
        palette = sns.color_palette('husl', 6)

        ax = sns.boxplot(
            x='Model',
            y=class_name,
            data=df,
            palette=palette,
            showfliers=False,
            order=model_order,
        )
        plt.ylabel('DSC', fontsize=36)
        plt.xticks(rotation=90, fontsize=30)
        plt.yticks(fontsize=30)
        ax.set_xlabel('')

        # Set y-limits for the current class
        ax.set_ylim(y_limits[class_name])

        sns.despine()
        plt.tight_layout()

        # Save the plot as a high-quality image file in PNG format
        save_path = os.path.join(save_dir, f'{class_name}_boxplot.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
