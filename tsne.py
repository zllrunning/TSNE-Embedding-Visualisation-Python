from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

from util.extract_feature_v1 import extract_feature
from backbone.model_irse import IR_50

def get_data():
    digits = datasets.load_digits(n_class=5)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    label = label.dtype(np.int8)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i][0]),
                 color=plt.cm.Set1(label[i][0] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def main():
    data = '/home/zll/data/face/data/for_tsne_vis'
    model = IR_50([112, 112])
    model_cp = '/home/zll/Downloads/code/FACE/face.evoLVe.PyTorch/checkpoint/backbone_ir50_ms1m_epoch120.pth'
    features, labels = extract_feature(data, model, model_cp, batch_size=32)
    # print(type(features), type(labels), features.shape, labels.shape)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(features)
    fig = plot_embedding(result, labels,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


if __name__ == '__main__':
    main()

