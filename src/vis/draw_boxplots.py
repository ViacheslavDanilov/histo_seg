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
    palette = sns.color_palette('bright', 6)

    # Iterate over columns except 'Model', 'Fold', and 'Epoch' to create boxplots
    classes = df.columns.difference(['Model', 'Fold', 'Epoch'])
    for class_name in classes:
        plt.figure(figsize=(8, 6))  # Adjust figure size for better presentation
        ax = sns.boxplot(x='Model', y=class_name, data=df, palette=palette)
        plt.ylabel('DSC')  # Add appropriate y-axis label
        ax.set_xlabel('')  # Remove x-axis label
        plt.title(f'Boxplot of {class_name} across Models')  # Add a descriptive title
        sns.despine()  # Remove the top and right spines for a cleaner look
        plt.tight_layout()

        # Save the plot as a high-quality image file in PNG format
        plt.savefig(
            os.path.join(save_dir, f'{class_name}_boxplot.png'),
            dpi=300,
            bbox_inches='tight',
        )
        plt.close()  # Close the current plot to release memory


if __name__ == '__main__':
    main()
