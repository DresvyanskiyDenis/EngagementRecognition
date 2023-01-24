import os
from functools import partial

import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from pytorch_utils.callbacks import TorchMetricEvaluator
import pandas as pd

from src.IWSDS2023.visual_subsystem.pose_subsystem.sequence_model.sequence_data_loader import SequenceDataLoader
from src.IWSDS2023.visual_subsystem.pose_subsystem.sequence_model.sequence_model import Seq2One_model
from src.IWSDS2023.visual_subsystem.pose_subsystem.sequence_model.sequence_model_training_cross_corpus import \
    load_cross_corpus_data


def validate_model(model, data_loader, device):
    metrics = {
        'accuracy': accuracy_score,
        'recall': partial(recall_score, average='macro'),
        'precision': partial(precision_score, average='macro'),
        'f1_score': partial(f1_score, average='macro')
    }


    model.eval()

    metric_evaluator = TorchMetricEvaluator(generator=data_loader,
                                            model=model,
                                            metrics=metrics,
                                            device=device,
                                            output_argmax=True,
                                            output_softmax=True,
                                            labels_argmax=True,
                                            loss_func=None)

    metric_results = metric_evaluator()
    for metric_name, metric_value in metric_results.items():
        print(f'{metric_name}: {metric_value}')
    s = str(metric_results['recall'])+','+str(metric_results['precision'])+','+str(metric_results['f1_score'])+','+str(metric_results['accuracy'])
    print(s)
    model.train()
    return s

def main(language:str):
    print('FOCAL LOSS!')
    print(language)
    # params
    BATCH_SIZE = 64
    test_params = pd.read_csv("/work/home/dsu/Model_weights/weights_of_best_models/sequence_to_one_experiments"\
                              "/Cross_corpus/Pose_model/%s/testing_params.csv"%language.capitalize())
    test_params = test_params.iloc[:19]
    path_to_all_weights = "/work/home/dsu/Model_weights/weights_of_best_models/sequence_to_one_experiments"\
                              "/Cross_corpus/Pose_model/%s/"%language.capitalize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load data
    train, dev, test = load_cross_corpus_data(language)
    #train = pd.read_csv("/work/home/dsu/IWSDS2023/NoXi_embeddings/All_languages/Pose_model/embeddings_train.csv")
    #dev = pd.read_csv("/work/home/dsu/IWSDS2023/NoXi_embeddings/All_languages/Pose_model/embeddings_dev.csv")
    #test = pd.read_csv("/work/home/dsu/IWSDS2023/NoXi_embeddings/All_languages/Pose_model/embeddings_test.csv")


    val_s = ''
    test_s = ''
    for index, row in test_params.iterrows():
        print('--------------------------------------------------------------------------------------')
        window_length = int(row['window_length'])
        weights_path = os.path.join(path_to_all_weights, row['ID']+ ".pt")
        print('The ID is:%s, window_length is:%s' % (row['ID'], window_length))
        # load model
        num_neurons = int(row['num_lstm_neurons'])
        num_layers = int(row['num_lstm_layers'])
        num_embeddings = 256
        model = Seq2One_model(input_shape=(BATCH_SIZE, window_length, num_embeddings), LSTM_neurons=tuple(num_neurons for i in range(num_layers)),
                          dropout=0.3,
                          dense_neurons=(256,),
                          dense_neurons_activation_functions=('tanh',), dense_dropout=None,
                          output_layer_neurons=5, output_activation_function='linear')
        model.load_state_dict(torch.load(weights_path))
        model.to(device)
        # create data loaders
        train_gen = SequenceDataLoader(dataframe=train, window_length=window_length, window_shift=window_length//2,
                                   labels_included=True, scaler="standard")
        scaler = train_gen.scaler
        dev_gen = SequenceDataLoader(dataframe=dev, window_length=window_length, window_shift=window_length//2,
                                 labels_included=True, scaler=scaler)
        test_gen = SequenceDataLoader(dataframe=test, window_length=window_length, window_shift=window_length // 2,
                                 labels_included=True, scaler=scaler)

        dev_gen = torch.utils.data.DataLoader(dev_gen, batch_size=BATCH_SIZE, num_workers=16, pin_memory=False)
        test_gen = torch.utils.data.DataLoader(test_gen, batch_size=BATCH_SIZE, num_workers=16, pin_memory=False)

        # validate model
        print('----------Development dataset----------')
        s_val = validate_model(model, dev_gen, device)
        val_s += s_val+ '\n'
        print('----------Test dataset----------')
        s_test = validate_model(model, test_gen, device)
        test_s += s_test + '\n'

    print('FINAL')
    print('dev')
    print(val_s)
    print('--------------test')
    print(test_s)


if __name__ == '__main__':
    main("french")