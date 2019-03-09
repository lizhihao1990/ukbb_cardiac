# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os, time, math
import numpy as np
import nibabel as nib
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from image_utils import *


""" Deployment parameters """
FLAGS = tf.app.flags.FLAGS
# tf.app.flags._global_parser.add_argument('--seq_name', choices=['sa', 'la_2ch', 'la_4ch', 'ao'],
#                                          default='sa', help="Sequence name.")
tf.app.flags.DEFINE_string('seq_name', 'sa',
                           "Sequence name.")
tf.app.flags.DEFINE_string('test_dir', '/vol/biomedic2/wbai/tmp/github/test',
                           'Path to the test set directory, under which images are organised in '
                           'subdirectories for each subject.')
tf.app.flags.DEFINE_string('dest_dir', '/vol/biomedic2/wbai/tmp/github/output',
                           'Path to the destination directory, where the segmentations will be saved.')
#tf.app.flags.DEFINE_string('model_path', '/vol/bitbucket/wbai/ukbb_cardiac/UKBB_2964/model/#FCN_sa_level5_filter16_22333_Adam_batch2_iter50000_lr0.001/#FCN_sa_level5_filter16_22333_Adam_batch2_iter50000_lr0.001.ckpt-50000',
#                           'Path to the saved trained model.')
tf.app.flags.DEFINE_string('model_path', '/vol/biomedic2/wbai/ukbb_cardiac/UKBB_2964/model/FCN_sa_level5_filter16_22333_Adam_batch2_iter50000_lr0.001/FCN_sa_level5_filter16_22333_Adam_batch2_iter50000_lr0.001.ckpt-50000',
                           'Path to the saved trained model.')
tf.app.flags.DEFINE_boolean('process_seq', True, "Process a time sequence of images.")
tf.app.flags.DEFINE_boolean('save_seg', True, "Save segmentation.")
tf.app.flags.DEFINE_boolean('clinical_measure', True, "Calculate clinical measures.")
tf.app.flags.DEFINE_boolean('cardiac_cnn_tf', False, "Using the previous model used in the paper.")
tf.app.flags.DEFINE_boolean('z_score', False, 'Normalise the image intensity to z-score. '
                                              'Otherwise, rescale the intensity.')
tf.app.flags.DEFINE_boolean('seg_4ch', False, 'Segment all the four chambers in long-axis view. '
                                              'The network is trained using Application 18545.')
tf.app.flags.DEFINE_boolean('upsample', False, 'Upsample the probability map and segmentation.')


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.train.import_meta_graph('{0}.meta'.format(FLAGS.model_path))
        saver.restore(sess, '{0}'.format(FLAGS.model_path))

        print('Start evaluating on the test set ...')
        start_time = time.time()

        # Process each subject subdirectory
        data_list = sorted(os.listdir(FLAGS.test_dir))
        processed_list = []
        table = []
        table_time = []
        for data in data_list:
            print(data)
            data_dir = os.path.join(FLAGS.test_dir, data)

            # if os.path.exists('{0}/seg_{1}.nii.gz'.format(data_dir, FLAGS.seq_name)):
            #     continue

            if FLAGS.process_seq:
                # Process the temporal sequence
                image_name = '{0}/{1}.nii.gz'.format(data_dir, FLAGS.seq_name)

                if not os.path.exists(image_name):
                    print('  Directory {0} does not contain an image with file name {1}. '
                          'Skip.'.format(data_dir, os.path.basename(image_name)))
                    continue

                dest_data_dir = os.path.join(FLAGS.dest_dir, data)
                if FLAGS.seg_4ch:
                    seg_name = '{0}/seg2_{1}.nii.gz'.format(dest_data_dir, FLAGS.seq_name)
                else:
                    seg_name = '{0}/seg_{1}.nii.gz'.format(dest_data_dir, FLAGS.seq_name)

                if os.path.exists(seg_name):
                    print('  Directory {0} already segmented. Skip.'.format(data_dir))
                    continue

                # Read the image
                print('  Reading {} ...'.format(image_name))
                nim = nib.load(image_name)
                image = nim.get_data()
                X, Y, Z, T = image.shape
                orig_image = image

                print('  Segmenting full sequence ...')
                start_seg_time = time.time()

                # Intensity normalisation
                if FLAGS.z_score:
                    image = normalise_intensity(image, 1.0)
                else:
                    image = rescale_intensity(image, (1.0, 99.0))

                # Prediction (segmentation)
                pred = np.zeros(image.shape)

                if FLAGS.upsample:
                    factor = 4
                    pred_zoom = np.zeros((X * factor, Y * factor, Z, T))

                # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                # in the network will result in the same image size at each resolution level.
                X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

                # Process each time frame
                for t in range(T):
                    # Transpose the shape to NXYC
                    image_fr = image[:, :, :, t]
                    image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                    image_fr = np.expand_dims(image_fr, axis=-1)

                    # Evaluate the network
                    if FLAGS.cardiac_cnn_tf:
                        prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'],
                                                    feed_dict={'image:0': image_fr})
                    else:
                        prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'],
                                                    feed_dict={'image:0': image_fr,
                                                               'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
                    pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                    pred[:, :, :, t] = pred_fr

                    if FLAGS.upsample:
                        prob_fr = np.transpose(prob_fr, axes=(1, 2, 0, 3))
                        prob_fr = prob_fr[x_pre:x_pre + X, y_pre:y_pre + Y]

                        nim2 = nib.Nifti1Image(prob_fr, nim.affine)
                        nim2.header['pixdim'] = nim.header['pixdim']
                        nib.save(nim2, '{0}/prob_{1}_fr{2:02d}.nii.gz'.format(dest_data_dir,
                                                                              FLAGS.seq_name, t))

                        # prob_fr_zoom = np.zeros((X * factor, Y * factor, Z, prob_fr.shape[-1]))
                        # for z in range(Z):
                        #     prob_fr_zoom[:, :, z, :] = ndimage.zoom(prob_fr[:, :, z, :],
                        #                                             zoom=(factor, factor, 1))
                        # pred_zoom[:, :, :, t] = np.argmax(prob_fr_zoom, axis=-1)

                seg_time = time.time() - start_seg_time
                print('  Segmentation time = {:3f}s'.format(seg_time))
                table_time += [seg_time]
                processed_list += [data]

                # ED frame defaults to be the first time frame.
                # Determine ES frame according to the minimum LV volume.
                if FLAGS.seq_name == 'sa' or FLAGS.seq_name == 'la_2ch' or FLAGS.seq_name == 'la_4ch':
                    k = {}
                    k['ED'] = 0
                    if FLAGS.seq_name == 'sa' or (FLAGS.seq_name == 'la_4ch' and FLAGS.seg_4ch):
                        k['ES'] = np.argmin(np.sum(pred == 1, axis=(0, 1, 2)))
                    else:
                        k['ES'] = np.argmax(np.sum(pred == 1, axis=(0, 1, 2)))
                    print('  ED frame = {:d}, ES frame = {:d}'.format(k['ED'], k['ES']))

                # Save the segmentation
                if FLAGS.save_seg:
                    print('  Saving segmentation ...')
                    dest_data_dir = os.path.join(FLAGS.dest_dir, data)
                    if not os.path.exists(dest_data_dir):
                        os.makedirs(dest_data_dir)

                    nim2 = nib.Nifti1Image(pred, nim.affine)
                    nim2.header['pixdim'] = nim.header['pixdim']
                    nib.save(nim2, seg_name)

                    # if FLAGS.upsample:
                    #     seg_zoom_name = '{0}/seg_up_{1}.nii.gz'.format(dest_data_dir, FLAGS.seq_name)
                    #     affine2 = np.dot(nim.affine,
                    #                      np.diag([1.0 / factor, 1.0/ factor, 1, 1]))
                    #     nim2 = nib.Nifti1Image(pred_zoom, affine2)
                    #     nim2.header['pixdim'] = nim.header['pixdim']
                    #     nib.save(nim2, seg_zoom_name)

                    if FLAGS.seq_name == 'sa' or FLAGS.seq_name == 'la_2ch' or FLAGS.seq_name == 'la_4ch':
                        for fr in ['ED', 'ES']:
                            if FLAGS.seq_name == 'la_4ch' and FLAGS.seg_4ch:
                                save_seg_name = '{0}/seg2_{1}_{2}.nii.gz'.format(dest_data_dir,
                                                                                 FLAGS.seq_name, fr)
                                nib.save(nib.Nifti1Image(pred[:, :, :, k[fr]], nim.affine),
                                         save_seg_name)
                            else:
                                save_image_name = '{0}/{1}_{2}.nii.gz'.format(dest_data_dir,
                                                                              FLAGS.seq_name, fr)
                                save_seg_name = '{0}/seg_{1}_{2}.nii.gz'.format(dest_data_dir,
                                                                                FLAGS.seq_name, fr)
                                nib.save(nib.Nifti1Image(orig_image[:, :, :, k[fr]], nim.affine),
                                         save_image_name)
                                nib.save(nib.Nifti1Image(pred[:, :, :, k[fr]], nim.affine),
                                         save_seg_name)

                # Evaluate the clinical measures
                if FLAGS.seq_name == 'sa' and FLAGS.clinical_measure:
                    print('  Evaluating clinical measures ...')
                    measure = {}
                    dx, dy, dz = nim.header['pixdim'][1:4]
                    volume_per_voxel = dx * dy * dz * 1e-3
                    density = 1.05

                    # Heart rate
                    duration_per_cycle = nim.header['dim'][4] * nim.header['pixdim'][4]
                    heart_rate = 60.0 / duration_per_cycle

                    for fr in ['ED', 'ES']:
                        measure[fr] = {}
                        measure[fr]['LVV'] = np.sum(pred[:, :, :, k[fr]] == 1) * volume_per_voxel
                        measure[fr]['LVM'] = np.sum(pred[:, :, :, k[fr]] == 2) * volume_per_voxel * density
                        measure[fr]['RVV'] = np.sum(pred[:, :, :, k[fr]] == 3) * volume_per_voxel

                    line = [measure['ED']['LVV'], measure['ES']['LVV'],
                            measure['ED']['LVM'],
                            measure['ED']['RVV'], measure['ES']['RVV'],
                            heart_rate]
                    table += [line]
            else:
                # Process ED and ES time frames
                image_ED_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ED')
                image_ES_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, 'ES')
                if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                    print('  Directory {0} does not contain an image with file name {1} or {2}. '
                          'Skip.'.format(data_dir, os.path.basename(image_ED_name),
                                          os.path.basename(image_ES_name)))
                    continue

                measure = {}
                for fr in ['ED', 'ES']:
                    image_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)

                    # Read the image
                    print('  Reading {} ...'.format(image_name))
                    nim = nib.load(image_name)
                    image = nim.get_data()
                    X, Y = image.shape[:2]
                    if image.ndim == 2:
                        image = np.expand_dims(image, axis=2)

                    print('  Segmenting {} frame ...'.format(fr))
                    start_seg_time = time.time()

                    # Intensity normalisation
                    if FLAGS.z_score:
                        image = normalise_intensity(image, 1.0)
                    else:
                        image = rescale_intensity(image, (1.0, 99.0))

                    # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                    # in the network will result in the same image size at each resolution level.
                    X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                    x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                    x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')

                    # Transpose the shape to NXYC
                    image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
                    image = np.expand_dims(image, axis=-1)

                    # Evaluate the network
                    if FLAGS.cardiac_cnn_tf:
                        prob, pred = sess.run(['prob:0', 'pred:0'],
                                              feed_dict={'image:0': image})
                    else:
                        prob, pred = sess.run(['prob:0', 'pred:0'],
                                              feed_dict={'image:0': image,
                                                         'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred = np.transpose(pred, axes=(1, 2, 0))
                    pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]

                    seg_time = time.time() - start_seg_time
                    print('  Segmentation time = {:3f}s'.format(seg_time))
                    table_time += [seg_time]

                    # Save the segmentation
                    if FLAGS.save_seg:
                        print('  Saving segmentation ...')
                        dest_data_dir = os.path.join(FLAGS.dest_dir, data)
                        if not os.path.exists(dest_data_dir):
                            os.makedirs(dest_data_dir)

                        nim2 = nib.Nifti1Image(pred, nim.affine)
                        nim2.header['pixdim'] = nim.header['pixdim']
                        if FLAGS.seg_4ch:
                            seg_name = '{0}/seg2_{1}_{2}.nii.gz'.format(dest_data_dir, FLAGS.seq_name, fr)
                        else:
                            seg_name = '{0}/seg_{1}_{2}.nii.gz'.format(dest_data_dir, FLAGS.seq_name, fr)
                        nib.save(nim2, seg_name)

                    # Evaluate the clinical measures
                    if FLAGS.seq_name == 'sa' and FLAGS.clinical_measure:
                        print('  Evaluating clinical measures ...')
                        dx, dy, dz = nim.header['pixdim'][1:4]
                        volume_per_voxel = dx * dy * dz * 1e-3
                        density = 1.05

                        measure[fr] = {}
                        measure[fr]['LVV'] = np.sum(pred == 1) * volume_per_voxel
                        measure[fr]['LVM'] = np.sum(pred == 2) * volume_per_voxel * density
                        measure[fr]['RVV'] = np.sum(pred == 3) * volume_per_voxel

                processed_list += [data]
                if FLAGS.clinical_measure and FLAGS.seq_name == 'sa':
                    line = [measure['ED']['LVV'], measure['ES']['LVV'],
                            measure['ED']['LVM'],
                            measure['ED']['RVV'], measure['ES']['RVV']]
                    table += [line]

        # Save the spreadsheet for the clinical measures
        if FLAGS.seq_name == 'sa' and FLAGS.clinical_measure:
            if FLAGS.process_seq:
                column_names = ['LVEDV (mL)', 'LVESV (mL)', 'LVM (g)', 'RVEDV (mL)', 'RVESV (mL)', 'Heart rate (bpm)']
            else:
                column_names = ['LVEDV (mL)', 'LVESV (mL)', 'LVM (g)', 'RVEDV (mL)', 'RVESV (mL)']
            df = pd.DataFrame(table, index=processed_list, columns=column_names)
            csv_name = os.path.join(FLAGS.dest_dir, '../clinical_measure.csv')
            print('  Saving clinical measures at {0} ...'.format(csv_name))
            df.to_csv(csv_name)

        if FLAGS.process_seq:
            print('Average segmentation time = {:.3f}s per sequence'.format(np.mean(table_time)))
        else:
            print('Average segmentation time = {:.3f}s per frame'.format(np.mean(table_time)))
        process_time = time.time() - start_time
        print('Including image I/O, CUDA resource allocation, '
              'it took {:.3f}s for processing {:d} subjects ({:.3f}s per subjects).'.format(
            process_time, len(processed_list), process_time / len(processed_list)))
