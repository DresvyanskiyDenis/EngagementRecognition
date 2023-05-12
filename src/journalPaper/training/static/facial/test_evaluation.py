import torch
from torchinfo import summary

from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1, Modified_EfficientNet_B4
from src.journalPaper.training.static import training_config
from src.journalPaper.training.static.data_preparation import load_data_and_construct_dataloaders
from src.journalPaper.training.static.model_evaluation import evaluate_model

if __name__=="__main__":
    # testing params
    model_type = 'EfficientNet-B1'
    batch_size = 32
    path_to_weights = ''


    # get generators, including test generator
    (train_generator, dev_generator, test_generator), class_weights = load_data_and_construct_dataloaders(
        path_to_data_NoXi=training_config.NOXI_FACIAL_PATH,
        path_to_data_DAiSEE=training_config.DAISEE_FACIAL_PATH,
        model_type=model_type,
        batch_size=batch_size,
        return_class_weights=True)

    # create and load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_type == "EfficientNet-B1":
        model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                         num_regression_neurons=None)
    elif model_type == "EfficientNet-B4":
        model = Modified_EfficientNet_B4(embeddings_layer_neurons=256, num_classes=training_config.NUM_CLASSES,
                                         num_regression_neurons=None)
    else:
        raise ValueError("Unknown model type: %s" % model_type)
    model = model.to(device)
    # print model architecture
    summary(model, (2, 3, training_config.MODEL_INPUT_SIZE[model_type], training_config.MODEL_INPUT_SIZE[model_type]))
    # load model weights
    model.load_state_dict(torch.load(path_to_weights))

    # test model
    test_metrics = evaluate_model(model, test_generator, device, metrics_name_prefix='test_', print_metrics=True)


