"""
How does only training on the motor imagery section of a trial affect our model?
"""

import numpy as np
import torch
from braindecode import EEGClassifier
from braindecode.datautil import infer_signal_properties
from braindecode.models import EEGNet, ShallowFBCSPNet
from braindecode.visualization import plot_confusion_matrix
from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from moabb.datasets import Cho2017
from moabb.paradigms import LeftRightImagery
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import Dataset, ValidSplit

from config import cache_config
from grad_cam import grad_cam
from perturbation import amplitude_perturbation_importance
from visualize import plot_loss_curve

resample_rate = 200.0
paradigm = LeftRightImagery(resample=resample_rate)

epochs, y, meta = paradigm.get_data(
    Cho2017(), cache_config=cache_config, return_epochs=True
)
X = epochs.get_data()
info = epochs.info

sig_props = infer_signal_properties(X, y, mode="classification")

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42)


# Splits on the samples as the stimulus is presented
stimuli_Xtrain, stimuli_Xtest = (
    Xtrain[:, :, int(resample_rate * 1.9 ) : int(resample_rate * 2.4)],
    Xtest[:, :, int(resample_rate * 1.9) : int(resample_rate * 2.4)],
)
# Splits on the samples 0.5s after stimulus is presented
motor_Xtrain, motor_Xtest = (
    Xtrain[:, :, int(resample_rate * 2.5) :],
    Xtest[:, :, int(resample_rate * 2.5) :],
)

le = LabelEncoder()
le.fit(y)
y_enc = le.transform(y)
ytrain_enc = le.transform(ytrain)
ytest_enc = le.transform(ytest)

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True

# scn_model = EEGNet(
#     n_chans=sig_props["n_chans"],
#     n_outputs=sig_props["n_outputs"],
#     n_times=sig_props["n_times"],
#     final_conv_length="auto",
#     F1=8,
#     D=2,
# )

# motor_eegnet_model = EEGNet(
#     n_chans=sig_props["n_chans"],
#     n_outputs=sig_props["n_outputs"],
#     n_times=motor_Xtrain.shape[2],
#     final_conv_length="auto",

#     F1=8,
#     D=2,
#     drop_prob=0.40
# )
stimuli_scn_model = ShallowFBCSPNet(
    n_chans=sig_props["n_chans"],
    n_outputs=sig_props["n_outputs"],
    n_times=stimuli_Xtrain.shape[2],
    final_conv_length="auto",
    drop_prob=0.6,
    chs_info=info["chs"],
    sfreq=resample_rate,
)
motor_scn_model = ShallowFBCSPNet(
    n_chans=sig_props["n_chans"],
    n_outputs=sig_props["n_outputs"],
    n_times=motor_Xtrain.shape[2],
    final_conv_length="auto",
    drop_prob=0.6,
    chs_info=info["chs"],
    sfreq=resample_rate,
)

# print(scn_model)
print(stimuli_scn_model)
print(motor_scn_model)

max_epochs = 300
batch_size = 64

lr = 0.001
weight_decay = 0

seed = 42

# scn_clf = EEGClassifier(
#     scn_model,
#     criterion=torch.nn.CrossEntropyLoss,
#     optimizer=torch.optim.AdamW,
#     train_split=ValidSplit(0.2, random_state=seed),
#     optimizer__lr=lr,
#     optimizer__weight_decay=weight_decay,
#     batch_size=batch_size,
#     max_epochs=max_epochs,
#     callbacks=[
#         (
#             "lr_scheduler",
#             LRScheduler(
#                 "ReduceLROnPlateau", monitor="valid_loss", patience=50, factor=0.5
#             ),
#         ),
#     ],
#     device=device,
#     classes=paradigm.events,
# )
stimuli_scn_clf = EEGClassifier(
    stimuli_scn_model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=ValidSplit(0.2, random_state=seed),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    max_epochs=max_epochs,
    callbacks=[
        # (
        #     "lr_scheduler",
        #     LRScheduler(
        #         "ReduceLROnPlateau", monitor="valid_loss", patience=20, factor=0.5
        #     ),
        # ),
        (
            "lr_scheduler",
            LRScheduler(
                "CosineAnnealingLR", monitor="valid_loss", T_max=max_epochs-1
            ),
        ),
    ],
    device=device,
    classes=paradigm.events,
)
motor_scn_clf = EEGClassifier(
    motor_scn_model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=ValidSplit(0.2, random_state=seed),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    max_epochs=max_epochs,
    callbacks=[
        (
            "lr_scheduler",
            LRScheduler(
                "CosineAnnealingLR", monitor="valid_loss", T_max=max_epochs-1
            ),
        ),
    ],
    device=device,
    classes=paradigm.events,
)


# scn_clf.fit(Xtrain, ytrain_enc)
TRAIN = True
if TRAIN:
    stimuli_scn_clf.fit(stimuli_Xtrain, ytrain_enc)
    motor_scn_clf.fit(motor_Xtrain, ytrain_enc)
    
    fig = plot_loss_curve([stimuli_scn_clf, motor_scn_clf], ["Stimuli", "Motor"])
    plt.title("Loss Curves of Models Trained on Motor vs Stimuli Intervals")
    plt.savefig("motor_vs_stimuli_loss_curve")
    plt.show()
    
    torch.save(stimuli_scn_model.state_dict(), "stimuli_scn_model_csn.pt")
    torch.save(motor_scn_model.state_dict(), "motor_scn_model_csn.pt")
else:
    stimuli_scn_model.load_state_dict(torch.load("stimuli_scn_model_csn.pt", weights_only=True))
    motor_scn_model.load_state_dict(torch.load("motor_scn_model_csn.pt", weights_only=True))
    
    stimuli_scn_model.eval()
    motor_scn_model.eval()
    
    stimuli_scn_clf.initialize()
    motor_scn_clf.initialize()


# print(f"ShallowConvNet on whole interval: {scn_clf.score(Xtest, ytest_enc)}")
print(
    f"ShallowConvNet on interval from 0.1s prior to cue until 0.4s after cue - test accuracy: {stimuli_scn_clf.score(stimuli_Xtest, ytest_enc)}"
)
print(
    f"ShallowConvNet on interval 0.5s after cue - test accuracy: {motor_scn_clf.score(motor_Xtest, ytest_enc)}"
)

stimuli_scn_confmat = confusion_matrix(ytest_enc, stimuli_scn_clf.predict(stimuli_Xtest))
plot_confusion_matrix(stimuli_scn_confmat, class_names=le.classes_)
plt.title("Stimuli Interval")
plt.savefig("stimuli_ShallowConvNet_confmat")
plt.show()

motor_scn_confmat = confusion_matrix(ytest_enc, motor_scn_clf.predict(motor_Xtest))
plot_confusion_matrix(motor_scn_confmat, class_names=le.classes_)
plt.title("Motor Imagery Interval")
plt.savefig("motor_ShallowConvNet_confmat")
plt.show()

left_idx = np.where(ytest_enc == 0)[0]
right_idx = np.where(ytest_enc == 1)[0]
# left_importances = []
# for i in left_idx:
#     importance, _, _ = grad_cam(
#         motor_scn_clf.module_,
#         torch.tensor(motor_Xtest[i], dtype=torch.float32).unsqueeze(0),
#         target_layer_name="conv_time_spat.conv_time",
#         class_idx=ytest_enc[i],
#         reduce_to="electrodes",
#     )
#     left_importances.append(importance)
# mean_left_importance = np.mean(left_importances, axis=0)
# plot_topomap(mean_left_importance, info, size=8, show=False)
# plt.savefig("motor_EEGNet_spatial_importance")


# Interpret stimuli model - left classes
stimuli_corr_left, freqs = amplitude_perturbation_importance(
    model=stimuli_scn_clf.module_,
    X=stimuli_Xtest[left_idx],
    class_idx=0,
    sfreq=resample_rate,
    n_iterations=30,
    noise_std=0.02,
    batch_size=batch_size,
    seed=seed,
)
signed_stimuli_corr_left = stimuli_corr_left.mean(axis=1)
plot_topomap(signed_stimuli_corr_left, info, size=8, show=False)
plt.title("Stimuli Interval Amplitude Correlation: Left")
plt.savefig("stimuli_ShallowConvNet_electrode_correlation_left")
plt.show()

# Interpret stimuli model - right classes
stimuli_corr_right, freqs = amplitude_perturbation_importance(
    model=stimuli_scn_clf.module_,
    X=stimuli_Xtest[right_idx],
    class_idx=1,
    sfreq=resample_rate,
    n_iterations=30,
    noise_std=0.02,
    batch_size=batch_size,
    seed=seed,
)
signed_stimuli_corr_right = stimuli_corr_right.mean(axis=1)
plot_topomap(signed_stimuli_corr_right, info, size=8, show=False)
plt.title("Stimuli Interval Amplitude Correlation: Right")
plt.savefig("stimuli_ShallowConvNet_electrode_correlation_right")
plt.show()

# Interpret motor model - left classes
motor_corr_left, freqs = amplitude_perturbation_importance(
    model=motor_scn_clf.module_,
    X=motor_Xtest[left_idx],
    class_idx=0,
    sfreq=resample_rate,
    n_iterations=30,
    noise_std=0.02,
    batch_size=batch_size,
    seed=seed,
)
signed_motor_corr_left = motor_corr_left.mean(axis=1)
plot_topomap(signed_motor_corr_left, info, size=8, show=False)
plt.title("Motor Interval Amplitude Correlation: Left")
plt.savefig("motor_ShallowConvNet_electrode_correlation_left")
plt.show()

# Interpret motor model - right classes
motor_corr_right, freqs = amplitude_perturbation_importance(
    model=motor_scn_clf.module_,
    X=motor_Xtest[right_idx],
    class_idx=1,
    sfreq=resample_rate,
    n_iterations=30,
    noise_std=0.02,
    batch_size=batch_size,
    seed=seed,
)
signed_motor_corr_right = motor_corr_right.mean(axis=1)
plot_topomap(signed_motor_corr_right, info, size=8, show=False)
plt.title("Motor Interval Amplitude Correlation: Right")
plt.savefig("motor_ShallowConvNet_electrode_correlation_right")
plt.show()
