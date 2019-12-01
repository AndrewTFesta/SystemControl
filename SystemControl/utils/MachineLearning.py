"""
@title
@description
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(y_true, y_pred, class_labels, title,
                          cmap: str = 'Blues', annotate_entries: bool = True, save_plot: str = None):
    style.use('ggplot')
    fig_title = f'Target: \'{title}\''
    conf_mat = confusion_matrix(y_true, y_pred)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    lower_bound = np.min(y_true) - 0.5
    upper_bound = np.max(y_true) + 0.5

    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    xtick_marks = np.arange(conf_mat.shape[1])
    ytick_marks = np.arange(conf_mat.shape[0])

    ax.set_xticks(xtick_marks)
    ax.set_yticks(ytick_marks)

    ax.set_xbound(lower=lower_bound, upper=upper_bound)
    ax.set_ybound(lower=lower_bound, upper=upper_bound)
    ax.invert_yaxis()

    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    ax.set_title(fig_title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    if annotate_entries:
        annot_format = '0.2f'
        thresh = conf_mat.max() / 2.
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                conf_entry = conf_mat[i, j]
                ax.text(
                    j, i, format(conf_entry, annot_format), ha='center', va='center',
                    color='white' if conf_entry > thresh else 'black'
                )
    fig.tight_layout()

    if save_plot:
        plt.savefig(save_plot)
    else:
        plt.show()
    plt.close()
    return


def split_data(data, targets, test_size=0.25, val_size=0.15):
    train_val_x, test_x, train_val_y, test_y = train_test_split(
        data, targets, test_size=test_size
    )

    train_x, val_x, train_y, val_y = train_test_split(
        train_val_x, train_val_y, test_size=val_size
    )

    data_dict = {
        'train': {'images': train_x, 'labels': train_y.to_numpy()},
        'validation': {'images': val_x, 'labels': val_y.to_numpy()},
        'test': {'images': test_x, 'labels': test_y.to_numpy()},
    }
    return data_dict
