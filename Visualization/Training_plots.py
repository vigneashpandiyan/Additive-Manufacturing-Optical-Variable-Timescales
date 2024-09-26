# -*- coding: utf-8 -*-
"""
@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch,vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"

@any reuse of this code should be authorized by the code author
"""
# %%
# Libraries to import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def training_plots(Training_loss, Training_accuracy, Training_loss_mean, Training_loss_std, learning_rate, Times, window_size, path):
    """
    Generate training plots for loss, accuracy, learning rate, and window lengths.

    Args:
        Training_loss (list): List of training loss values.
        Training_accuracy (list): List of training accuracy values.
        Training_loss_mean (list): List of mean training loss values.
        Training_loss_std (list): List of training loss standard deviation values.
        learning_rate (list): List of learning rate values.
        Times (list): List of time values.
        window_size (int): Size of the window.

    Returns:
        None
    """

    Training_loss = np.asarray(Training_loss)
    Training_loss = Training_loss.astype(np.float64)
    classfile = path+'/'+'Training_loss'+'.npy'
    np.save(classfile, Training_loss, allow_pickle=True)

    Training_accuracy = np.asarray(Training_accuracy)
    Training_accuracy = Training_accuracy.astype(np.float64)
    classfile = path+'/'+'Training_accuracy'+'.npy'
    np.save(classfile, Training_accuracy, allow_pickle=True)

    learning_rate = np.asarray(learning_rate)
    learning_rate = learning_rate.astype(np.float64)
    classfile = path+'/'+'learning_rate'+'.npy'
    np.save(classfile, learning_rate, allow_pickle=True)

    Times = np.asarray(Times)
    Times = Times.astype(np.float64)
    classfile = path+'/'+'Times'+'.npy'
    np.save(classfile, Times, allow_pickle=True)

    fig = plt.figure(figsize=(5, 3))
    Times = [window_size - x for x in Times]
    plt.plot(Times, 'blue')
    plt.title('Window lengths')
    plt.xlabel("Epoch")
    plt.ylabel("Window lengths")
    plt.ylim(500, 1500)
    plt.legend(labels=['Window lengths'], loc='best',
               fontsize=15, frameon=False)
    plotname = path+'/'+'Window lengths'+'.png'
    plt.savefig(plotname, bbox_inches='tight', dpi=200)
    plt.show()

    fig = plt.figure(figsize=(5, 3))
    plt.plot(Training_loss, 'blue')
    plt.title('Training loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(labels=['Training loss'], loc='best',
               fontsize=15, frameon=False)
    plotname = path+'/'+'Training_loss'+'.png'
    plt.savefig(plotname, bbox_inches='tight', dpi=200)
    plt.show()

    fig = plt.figure(figsize=(5, 3))
    plt.plot(Training_accuracy, 'red')
    plt.title('Training accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(labels=['Training accuracy'],
               loc='best', fontsize=15, frameon=False)
    plotname = path+'/'+'Training_accuracy'+'.png'
    plt.savefig(plotname, bbox_inches='tight', dpi=200)
    plt.show()

    fig = plt.figure(figsize=(5, 3))
    plt.plot(learning_rate, 'black')
    plt.title('learning_rate')
    plt.xlabel("Epoch")
    plt.ylabel("learningrate")
    plt.legend(labels=['Learning rate'], loc='best',
               fontsize=15, frameon=False)
    plotname = path+'/'+'learning_rate'+'.png'
    plt.savefig(plotname, bbox_inches='tight', dpi=200)
    plt.show()

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(6, 3))
    Loss_value = pd.DataFrame(Training_loss)
    Training_loss_mean = pd.DataFrame(Training_loss_mean)
    Training_loss_std = pd.DataFrame(Training_loss_std)

    under_line = (Training_loss_mean - Training_loss_std)[0]
    over_line = (Training_loss_mean + Training_loss_std)[0]
    fig, ax = plt.subplots(figsize=(6, 3))
    plt.plot(Loss_value, 'red', linewidth=2.0, label='Training Loss')
    plt.fill_between(
        Training_loss_std.index,
        under_line,
        over_line,
        alpha=.650,
        label='Loss Std.',
        color='b'
    )
    plt.title('Epochs vs Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plot_1 = path+'/'+' Average_Loss_Circle' + '.png'
    plt.savefig(plot_1, dpi=600, bbox_inches='tight')
    plt.show()
