import gc
import glob
import os
import sys
sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])



from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.HRnet_training_wandb import get_data_loaders_from_data
from functools import partial

import torch
import torchvision.transforms as T
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

from pytorch_utils.callbacks import TorchMetricEvaluator
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.HRNet import load_HRNet_model, modified_HRNet
from src.NoXi.visual_subsystem.pose_subsystem.frame_wise_model.utils import load_NoXi_data_all_languages, \
    convert_image_to_float_and_scale, load_NoXi_data_cross_corpus


def _validate_model_on_dataset(dataset:torch.utils.data.DataLoader, model:torch.nn.Module, device):
    # specify val metrics
    val_metrics = {
        'recall': partial(recall_score, average='macro'),
        'precision': partial(precision_score, average='macro'),
        'f1_score:': partial(f1_score, average='macro'),
        'confusion_matrix': confusion_matrix
    }
    # make evaluator, which will be used to evaluate the model
    metric_evaluator = TorchMetricEvaluator(generator = dataset,
                 model=model,
                 metrics=val_metrics,
                 device=device,
                 output_argmax=True,
                 output_softmax=True,
                 labels_argmax=True,
                 loss_func=None)

    # evaluate the model
    results = metric_evaluator()

    return results



def main():
    language = "german"
    print("Start...")
    # params
    BATCH_SIZE = 64
    NUM_CLASSES = 5
    path_to_dir_with_weights = "/work/home/dsu/Model_weights/weights_of_best_models/frame_to_frame_experiments/Pose_model/Cross_corpus/"\
                               +language.capitalize()+"/"
    paths_to_weights= glob.glob(os.path.join(path_to_dir_with_weights, "*.pt"))
    # specify device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load data
    train, dev, test = load_NoXi_data_cross_corpus(test_corpus=language, train_labels_as_categories=False,
                                                    dev_labels_as_categories=False,test_labels_as_categories=False)
    train_gen, dev_gen, test_gen = get_data_loaders_from_data(train, dev, test, augment=True, augment_prob=0.05,
                                                              batch_size=BATCH_SIZE,
                                                              preprocessing_functions=[T.Resize(size=(256, 256)),
                                                                                       convert_image_to_float_and_scale,
                                                                                       T.Normalize(
                                                                                           mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.225])
                                                                                       ])  # From HRNet



    for path_to_weights in paths_to_weights:
        # create model and load its weights
        HRNet = load_HRNet_model(device="cuda" if torch.cuda.is_available() else "cpu",
                                 path_to_weights="/work/home/dsu/simpleHRNet/pose_hrnet_w32_256x192.pth")
        model = modified_HRNet(HRNet, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(path_to_weights))
        model.to(device)

        # calculate metrics
        val_metrics = _validate_model_on_dataset(dataset=dev_gen, model=model, device=device)
        test_metrics = _validate_model_on_dataset(dataset=test_gen, model=model, device=device)
        # print results
        print("HRNet model, weights: {}".format(path_to_weights))
        print("Validation metrics:")
        for key, value in val_metrics.items():
            print(key, value)
        print("Test metrics:")
        for key, value in test_metrics.items():
            print(key, value)
        print("-----------------------------------------------------")
        # clear RAM
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Cross corpus model evaluation...")
    main()
