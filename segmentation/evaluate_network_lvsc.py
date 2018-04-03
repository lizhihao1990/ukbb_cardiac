import os
import time
import cv2
import numpy as np
import nibabel as nib
import pandas as pd


def np_categorical_dice(pred, truth, k):
    # Dice overlap metric for label value k
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


def apd(A, B, unit):
    # The average perpendicular distance measures the distance from the automatically segmented contour to
    # the corresponding manually drawn expert contour, averaged over all contour points.
    # Find contours and retrieve all the points
    A = A.astype('uint8')
    _, contours, _ = cv2.findContours(cv2.inRange(A, 1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts_A = contours[0]
    for i in range(1, len(contours)):
        pts_A = np.vstack((pts_A, contours[i]))

    B = B.astype('uint8')
    _, contours, _ = cv2.findContours(cv2.inRange(B, 1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts_B = contours[0]
    for i in range(1, len(contours)):
        pts_B = np.vstack((pts_B, contours[i]))

    # Distance matrix between point sets
    M = np.zeros((len(pts_A), len(pts_B)))
    for i in range(len(pts_A)):
        for j in range(len(pts_B)):
            M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

    return 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * unit


if __name__ == '__main__':
    for iteration in range(1000, 11000, 1000):
        print(iteration)
        data_base = '/vol/bitbucket/wbai/ukbb_cardiac/LVSC_2009/seg_LVSC_2009_fine_tune_iter{0}'.format(iteration)

        for dataset in ['challenge_validation']:
            data_path = os.path.join(data_base, dataset)
            data_list = sorted(os.listdir(data_path))

            # Evaluate endo and epicardium Dice and APD
            table = []
            eids = []
            for data in data_list:
                data_dir = os.path.join(data_path, data)

                dice = {}
                for fr in ['ED', 'ES']:
                    dice[fr] = {}

                    # Manual segmentation
                    nim = nib.load(os.path.join(data_dir, 'label_{0}.nii.gz'.format(fr)))
                    dx, dy = nim.header['pixdim'][1:3]
                    truth = np.squeeze(nim.get_data())

                    # Automatic segmentation
                    nim = nib.load(os.path.join(data_dir, 'seg_sa_{0}.nii.gz'.format(fr)))
                    pred = np.squeeze(nim.get_data())

                    dice[fr]['LV'] = np_categorical_dice(pred, truth, 1)
                    dice[fr]['Myo'] = np_categorical_dice(pred, truth, 2)

                line = [dice['ED']['LV'], dice['ED']['Myo'],
                        dice['ES']['LV']]
                table += [line]
                eids += [data]

            print(np.mean(table, axis=0))
            print(np.std(table, axis=0))
            column_names = ['dice_lved', 'dice_myoed',
                            'dice_lves']
            df = pd.DataFrame(table, index=eids, columns=column_names)
            df.to_csv(os.path.join(data_base, 'eval_{0}.csv'.format(dataset)))
