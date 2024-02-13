import gc
import os
from csv import DictWriter
from glob import glob

import cv2
import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.utils import CLASS_ID_REVERSED
from src.models.smp.model import HistologySegmentationModel


def to_tensor(
    x: np.ndarray,
) -> np.ndarray:
    return x.transpose([2, 0, 1]).astype('float32')


get_tensor = torchvision.transforms.ToTensor()


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='inference_smp_model',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    model = HistologySegmentationModel.load_from_checkpoint(
        f'models/{cfg.model_name}/ckpt/best.ckpt',
        arch=cfg.architecture,
        encoder_name=cfg.encoder,
        model_name=cfg.model_name,
        in_channels=3,
        classes=cfg.classes,
        lr=0.0001,
        optimizer_name='Adam',
        save_img_per_epoch=False,
    )
    model.eval()

    with open(f'models/{cfg.model_name}/metrics_eval.csv', 'a', newline='') as f_object:
        fieldnames = [
            'Threshold',
            'Class',
            'TP',
            'FP',
            'TN',
            'FN',
            'Precision',
            'Recall',
            'Sensitivity',
            'Specificity',
            'TPR',
            'FPR',
            'F1',
        ]
        writer = DictWriter(f_object, fieldnames=fieldnames)
        writer.writeheader()

        masks_gr, masks_predict = [], []

        for idy, img_path in enumerate(glob(f'{cfg.data_dir}/*.[pj][np][g]')[:4]):
            image = cv2.imread(img_path)
            image = cv2.resize(image, (cfg.input_size, cfg.input_size))
            mask_path = img_path.replace('img', 'mask')
            mask_path = mask_path.replace('jpg', 'png')
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(
                mask,
                (cfg.input_size, cfg.input_size),
                interpolation=cv2.INTER_NEAREST,
            )

            masks = [(mask == v) for v in CLASS_ID_REVERSED]
            masks = np.stack(masks, axis=-1).astype('float')
            image = to_tensor(np.array(image))

            mask_predict = model(torch.Tensor([image]).to('cuda')).cpu().detach()
            mask_predict = mask_predict.sigmoid()

            masks_predict.append(mask_predict.to('cpu'))
            masks_gr.append(masks)

            gc.collect()
            torch.cuda.empty_cache()

        for th in tqdm(range(1, 100, 1)):
            th *= 0.01
            tp, fp, fn, tn = None, None, None, None

            for masks, mask_pred in zip(masks_gr, masks_predict):
                mask_pred = (mask_pred >= th).float()

                tp_, fp_, fn_, tn_ = smp.metrics.get_stats(
                    mask_pred.long(),
                    torch.Tensor([to_tensor(np.array(masks))]).long(),
                    mode='multilabel',
                    num_classes=len(cfg.classes),
                )
                if tp is None:
                    tp, fp, fn, tn = tp_, fp_, fn_, tn_
                else:
                    tp += tp_
                    fp += fp_
                    fn += fn_
                    tn += tn_

            f1 = smp.metrics.f1_score(tp, fp, fn, tn).cpu().numpy()[0]
            precision = smp.metrics.precision(tp, fp, fn, tn).cpu().numpy()[0]
            recall = smp.metrics.recall(tp, fp, fn, tn).cpu().numpy()[0]
            sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn).cpu().numpy()[0]
            specificity = smp.metrics.specificity(tp, fp, fn, tn).cpu().numpy()[0]
            tp = tp.cpu().numpy()[0]
            fp = fp.cpu().numpy()[0]
            fn = fn.cpu().numpy()[0]
            tn = tn.cpu().numpy()[0]

            for class_id in CLASS_ID_REVERSED:
                writer.writerow(
                    {
                        'Threshold': th,
                        'Class': CLASS_ID_REVERSED[class_id],
                        'TP': tp[class_id - 1],
                        'FP': fp[class_id - 1],
                        'TN': tn[class_id - 1],
                        'FN': fn[class_id - 1],
                        'Precision': precision[class_id - 1],
                        'Recall': recall[class_id - 1],
                        'Sensitivity': sensitivity[class_id - 1],
                        'Specificity': specificity[class_id - 1],
                        'TPR': tp[class_id - 1] / (tp[class_id - 1] + fn[class_id - 1]),
                        'FPR': fp[class_id - 1] / (fp[class_id - 1] + tn[class_id - 1]),
                        'F1': f1[class_id - 1],
                    },
                )
        f_object.close()

    # df = pd.read_csv(f'models/{cfg.model_name}/metrics_eval.csv')
    #
    # for class_name in CLASS_ID:
    #     df_class = df.loc[df.Class == class_name]
    #     print(f'{class_name} AUC: {metrics.auc(df_class.FPR, df_class.TPR)}')


if __name__ == '__main__':
    main()
