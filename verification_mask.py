"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
from scipy import interpolate
import datetime
import bcolz
import os
import natsort
from PIL import Image
import tqdm
import imp
import time
import torch
from torchvision import transforms
from backbone.model_irse import IR_101, l2_norm
from network import Network
from evaluation import metrics


def pair_euc_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    x1, x2 = np.array(x1), np.array(x2)
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist


def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert sigma_sq2 is None, 'either pass in concated features, or mu, sigma_sq for both!'
        D = int(x1.shape[1] / 2)
        mu1, sigma_sq1 = x1[:,:D], x1[:,D:]
        mu2, sigma_sq2 = x2[:,:D], x2[:,D:]
    else:
        x1, x2 = np.array(x1), np.array(x2)
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)
    return -dist


def calculate_roc(thresholds
                  , embeddings1
                  , embeddings2
                  , actual_issame
                  , nrof_folds=10
                  , pca=0
                  , compare_func=pair_euc_score
                  , sigma_sq1=None
                  , sigma_sq2=None):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        # diff = np.subtract(embeddings1, embeddings2)
        # dist = np.sum(np.square(diff), 1)
        dist = compare_func(embeddings1, embeddings2, sigma_sq1, sigma_sq2)
        print(dist)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds
                  , embeddings1
                  , embeddings2
                  , actual_issame
                  , far_target
                  , nrof_folds=10
                  , compare_func=pair_euc_score
                  , sigma_sq1=None
                  , sigma_sq2=None):
    '''
    Copy from [insightface](https://github.com/deepinsight/insightface)
    :param thresholds:
    :param embeddings1:
    :param embeddings2:
    :param actual_issame:
    :param far_target:
    :param nrof_folds:
    :return:
    '''
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    dist = compare_func(embeddings1, embeddings2, sigma_sq1, sigma_sq2)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_acc(embeddings1
                  , embeddings2
                  , actual_issame
                  , nrof_folds=10
                  , compare_func=pair_euc_score
                  , sigma_sq1=None
                  , sigma_sq2=None):

    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = nrof_folds
    accuracies = np.zeros(nrof_folds, dtype=np.float32)
    thresholds = np.zeros(nrof_folds, dtype=np.float32)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    dist = compare_func(embeddings1, embeddings2, sigma_sq1, sigma_sq2)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Training
        _, thresholds[fold_idx] = metrics.accuracy(dist[train_set], actual_issame[train_set])

        # Testing
        accuracies[fold_idx], _ = metrics.accuracy(dist[test_set], actual_issame[test_set], np.array([thresholds[fold_idx]]))

    accuracy = np.mean(accuracies)
    threshold = - np.mean(thresholds)
    return accuracy, threshold


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame) + 1e-8
    n_diff = np.sum(np.logical_not(actual_issame)) + 1e-8
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0, compare_func=pair_euc_score, sigma_sq=None):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    sigma_sq1 = None
    sigma_sq2 = None
    if sigma_sq is not None:
        sigma_sq1 = sigma_sq[0::2]
        sigma_sq2 = sigma_sq[1::2]

    accuracy, threshold = calculate_acc(embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds=nrof_folds
                                        , compare_func=compare_func, sigma_sq1=sigma_sq1, sigma_sq2=sigma_sq2)
    # tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
    #                                    np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca
    #                                    , compare_func=compare_func, sigma_sq1=sigma_sq1, sigma_sq2=sigma_sq2)
    # thresholds = np.arange(0, 4, 0.001)
    # val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
    #                                   np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds
    #                                   , compare_func=compare_func, sigma_sq1=sigma_sq1, sigma_sq2=sigma_sq2)
    return accuracy, threshold


def data_iter(datasets, batch_size):
    data_num = datasets.shape[0]
    for i in range(0, data_num, batch_size):
        yield datasets[i:min(i+batch_size, data_num), ...]


def test(data_set, sess, embedding_tensor, batch_size, label_shape=None, feed_dict=None, input_placeholder=None):
    '''
    referenc official implementation [insightface](https://github.com/deepinsight/insightface)
    :param data_set:
    :param sess:
    :param embedding_tensor:
    :param batch_size:
    :param label_shape:
    :param feed_dict:
    :param input_placeholder:
    :return:
    '''
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        datas = data_list[i]
        embeddings = None
        feed_dict.setdefault(input_placeholder, None)
        for idx, data in enumerate(data_iter(datas, batch_size)):
            data_tmp = data.copy()    # fix issues #4
            data_tmp -= 127.5
            data_tmp *= 0.0078125
            feed_dict[input_placeholder] = data_tmp
            time0 = datetime.datetime.now()
            _embeddings = sess.run(embedding_tensor, feed_dict)
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((datas.shape[0], _embeddings.shape[1]))
            try:
                embeddings[idx*batch_size:min((idx+1)*batch_size, datas.shape[0]), ...] = _embeddings
            except ValueError:
                print('idx*batch_size value is %d min((idx+1)*batch_size, datas.shape[0]) %d, batch_size %d, data.shape[0] %d' %
                      (idx*batch_size, min((idx+1)*batch_size, datas.shape[0]), batch_size, datas.shape[0]))
                print('embedding shape is ', _embeddings.shape)
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            # print(_em.shape, _norm)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def ver_test(ver_list, ver_name_list, nbatch, sess, embedding_tensor, batch_size, feed_dict, input_placeholder):
    results = []
    for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = test(data_set=ver_list[i], sess=sess, embedding_tensor=embedding_tensor,
                                                              batch_size=batch_size, feed_dict=feed_dict,
                                                              input_placeholder=input_placeholder)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
    return results


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def read_img(img, transform):
    pil_image = Image.open(os.path.join(img))
    image_tensor = transform(pil_image)
    return image_tensor


def generate_pair(pair_list_path, file_dir):
    pair_list = open(pair_list_path, "r")
    pair_list = [line.rstrip() for line in pair_list]

    pair_flag_list = []
    dirFiles_list = []
    for pair_line in pair_list:
        pair_line_list = pair_line.split(' ')
        filepath1 = file_dir + pair_line_list[0]
        filepath2 = file_dir + pair_line_list[1]
        if os.path.isfile(filepath1) and os.path.isfile(filepath2):
            dirFiles_list.append(filepath1)
            dirFiles_list.append(filepath2)
            is_same = True if pair_line_list[2] == '1' else False
            pair_flag_list.append(is_same)

    return pair_flag_list, dirFiles_list


def eval(data_path, file_name):
    pairs, dirFiles = generate_pair(data_path, file_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 1
    INPUT_SIZE = [112, 112]
    backbone_path = './backbone/CurricularFace_Backbone.pth'
    backbone_model = IR_101(INPUT_SIZE).to(device)

    checkpoint = torch.load(backbone_path, map_location=lambda storage, loc: storage)
    if 'state_dict' not in checkpoint:
        backbone_model.load_state_dict(checkpoint)
    else:
        print('use dict')
        pretrained_weights = checkpoint['state_dict']
        model_weights = backbone_model.state_dict()
        pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                              if k in model_weights}
        model_weights.update(pretrained_weights)
        backbone_model.load_state_dict(model_weights)

    backbone_model.eval()

    # Load model files and config file
    config_file = 'config/sphere64_msarcface.py'
    config = imp.load_source('config', config_file)
    config.batch_format['size'] = 1
    network = Network()
    network.initialize(config)
    network.load_model('log/sphere64_msarcface_am_PFE/20201024-090224')

    data_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE[0], INPUT_SIZE[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    embeddings_org_list = []
    embeddings_mu_list = []
    embeddings_sigma_list = []

    for i, (filename) in tqdm.tqdm(enumerate((dirFiles))):
        start_time = time.time()
        img_tensor = read_img(filename, data_transform)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            mu, conv_final = backbone_model(img_tensor)
            mu = l2_norm(mu)
            mu = mu.cpu().data.numpy()
            conv_final = conv_final.cpu().data.numpy()

        embeddings_org_list.append(mu.reshape(-1))
        mu, sigma_sq = network.extract_feature(mu, conv_final)

        # print('mu:', mu.shape)
        # print('sigma_sq:', sigma_sq.shape)

        embeddings_mu_list.append(mu.reshape(-1))
        embeddings_sigma_list.append(sigma_sq.reshape(-1))

    embeddings_org = np.array(embeddings_org_list)
    accuracy, threshold = evaluate(embeddings_org, pairs, nrof_folds=10)
    print('Org Accuracy: %1.5f' % accuracy)
    print('threshold: %1.5f' % threshold)

    embeddings_mu = np.array(embeddings_mu_list)
    embeddings_sigma = np.array(embeddings_sigma_list)

    accuracy, threshold = evaluate(embeddings_mu, pairs, nrof_folds=10, compare_func=pair_MLS_score,
                                   sigma_sq=embeddings_sigma)
    print('MLS Accuracy: %1.5f' % accuracy)
    print('threshold: %1.5f' % threshold)


def main():
    pair_list_path = '../face_dataset/masked_pairs.txt'
    img_path = '../face_dataset/masked_whn_crop/'
    eval(pair_list_path, img_path)


if __name__ == "__main__":
    main()