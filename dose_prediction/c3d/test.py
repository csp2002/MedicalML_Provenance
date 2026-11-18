# -*- encoding: utf-8 -*-
# Copied from RTDosePrediction (https://github.com/yangsenwxy/RTDosePrediction), Src/C3D/test.py.
# Changed: reads the volumes of preprocess/csv_to_nii.py; arguments for run / data / output folders and split;
# scores with common/evaluate_openkbp.py and writes metrics.json and per_patient.csv.
"""Predict the test patients with a trained C3D model and score them with the OpenKBP metrics.

    cd dose_prediction/c3d
    python test.py --run_dir ../runs/c3d_bs2_it80000_<timestamp> --tta     # best_val_evaluation_index.pkl
    python test.py --ckpt /path/to/C3D_bs4_iter80000.pkl --out_dir ../runs/eval_c3d_published --tta
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

from evaluate_openkbp import evaluate  # noqa: E402
from model import *  # noqa: E402,F403
from NetworkTrainer.network_trainer import *  # noqa: E402,F403


def read_data(patient_dir):
    dict_images = {}
    list_structures = ['CT',
                       'PTV70',
                       'PTV63',
                       'PTV56',
                       'possible_dose_mask',
                       'Brainstem',
                       'SpinalCord',
                       'RightParotid',
                       'LeftParotid',
                       'Esophagus',
                       'Larynx',
                       'Mandible']

    for structure_name in list_structures:
        file_name = 'ct' if structure_name == 'CT' else structure_name
        structure_file = patient_dir + '/' + file_name + '.nii.gz'

        if structure_name == 'CT':
            dtype = sitk.sitkInt16
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            dict_images[structure_name] = sitk.GetArrayFromImage(dict_images[structure_name])[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

    dict_images['CT'] = dict_images['CT'] - 1024  # HU
    return dict_images


def pre_processing(dict_images):
    # PTVs
    PTVs = 70.0 / 70. * dict_images['PTV70'] \
           + 63.0 / 70. * dict_images['PTV63'] \
           + 56.0 / 70. * dict_images['PTV56']

    # OARs
    list_OAR_names = ['Brainstem',
                      'SpinalCord',
                      'RightParotid',
                      'LeftParotid',
                      'Esophagus',
                      'Larynx',
                      'Mandible'
                      ]
    OAR_all = np.concatenate([dict_images[OAR_name] for OAR_name in list_OAR_names], axis=0)

    # CT image
    CT = dict_images['CT']
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = CT.astype(np.float32) / 1000.

    # Possible mask
    possible_dose_mask = dict_images['possible_dose_mask']

    list_images = [np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
                   possible_dose_mask]
    return list_images


def copy_sitk_imageinfo(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2


# Input is C*Z*H*W
def flip_3d(input_, list_axes):
    if 'Z' in list_axes:
        input_ = input_[:, ::-1, :, :]
    if 'W' in list_axes:
        input_ = input_[:, :, :, ::-1]

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    list_prediction_B = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_3d(input_.copy(), list_flip_axes)
        augmented_input = torch.from_numpy(augmented_input.astype(np.float32))
        augmented_input = augmented_input.unsqueeze(0).to(trainer.setting.device)
        [_, prediction_B] = trainer.setting.network(augmented_input)

        # Aug back to original order
        prediction_B = flip_3d(np.array(prediction_B.cpu().data[0, :, :, :, :]), list_flip_axes)

        list_prediction_B.append(prediction_B[0, :, :, :])

    return np.mean(list_prediction_B, axis=0)


def inference(trainer, list_patient_dirs, save_path, do_TTA=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in tqdm(list_patient_dirs):
            patient_id = patient_dir.split('/')[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]
            possible_dose_mask = list_images[1]

            # Test-time augmentation
            if do_TTA:
                TTA_mode = [[], ['Z'], ['W'], ['Z', 'W']]
            else:
                TTA_mode = [[]]
            prediction = test_time_augmentation(trainer, input_, TTA_mode)

            # Pose-processing
            prediction[np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)] = 0
            prediction = 70. * prediction

            # Save prediction to nii image
            templete_nii = sitk.ReadImage(patient_dir + '/possible_dose_mask.nii.gz')
            prediction_nii = sitk.GetImageFromArray(prediction)
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
            if not os.path.exists(save_path + '/' + patient_id):
                os.mkdir(save_path + '/' + patient_id)
            sitk.WriteImage(prediction_nii, save_path + '/' + patient_id + '/dose.nii.gz')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', default=None, help='output folder of train.py')
    parser.add_argument('--ckpt', default='best_val_evaluation_index.pkl',
                        help='file name inside --run_dir, or a full path')
    parser.add_argument('--data_dir', default=os.path.join(HERE, '..', 'data', 'preprocessed'))
    parser.add_argument('--split', default='test', choices=['validation', 'test'])
    parser.add_argument('--out_dir', default=None, help='default: <run_dir>/eval_<split>_<ckpt>[_tta]')
    parser.add_argument('--tta', action='store_true', help='test-time augmentation (4 flips)')
    parser.add_argument('--patients', nargs='+', default=None, help='only these patients (default: the whole split)')
    parser.add_argument('--GPU_id', type=int, default=0, help='GPU id used for testing (default: 0; -1: CPU)')
    args = parser.parse_args()

    ckpt = args.ckpt if os.path.isabs(args.ckpt) or args.run_dir is None else os.path.join(args.run_dir, args.ckpt)
    tag = 'eval_%s_%s%s' % (args.split, os.path.splitext(os.path.basename(ckpt))[0], '_tta' if args.tta else '')
    out_dir = args.out_dir or os.path.join(args.run_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    trainer = NetworkTrainer()
    trainer.setting.project_name = 'C3D'
    trainer.setting.output_dir = out_dir

    trainer.setting.network = Model(in_ch=9, out_ch=1,
                                    list_ch_A=[-1, 16, 32, 64, 128, 256],
                                    list_ch_B=[-1, 32, 64, 128, 256, 512])

    # Load model weights
    trainer.init_trainer(ckpt_file=ckpt,
                         list_GPU_ids=[args.GPU_id],
                         only_network=True)

    # Start inference
    print('\n\n# Start inference !')
    case_root = os.path.join(args.data_dir, args.split + '-pats')
    patients = args.patients or sorted((p for p in os.listdir(case_root) if p.startswith('pt_')),
                                       key=lambda p: int(p.split('_')[1]))
    inference(trainer, [os.path.join(case_root, p) for p in patients], save_path=os.path.join(out_dir, 'predictions'),
              do_TTA=args.tta)

    # Evaluation
    print('\n\n# Start evaluation !')
    summary, rows = evaluate(os.path.join(out_dir, 'predictions'), case_root, patients)
    summary.update(checkpoint=os.path.abspath(ckpt), split=args.split, tta=args.tta)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'per_patient.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient', 'dose_mae', 'dvh_mae'])
        writer.writeheader()
        writer.writerows(rows)

    print('\n\nDose score is: ' + str(summary['dose_score']))
    print('DVH score is: ' + str(summary['dvh_score']))
