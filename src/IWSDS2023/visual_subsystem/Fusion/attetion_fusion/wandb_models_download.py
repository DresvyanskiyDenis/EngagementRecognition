import sys
from typing import Optional

sys.path.extend(["/work/home/dsu/datatools/"])
sys.path.extend(["/work/home/dsu/engagement_recognition_project_server/"])
sys.path.extend(["/work/home/dsu/simpleHRNet/"])


import wandb
import os
import pandas as pd
import numpy as np

from wandb_utils.sweep_information import get_top_n_sweep_runs, get_config_info_about_runs, download_model_from_run, \
    get_sweep_info


def download_models_from_sweep(sweep_id:str, top_n:int, metric:str, output_path:str, model_name:str,
                               new_model_name_prefix:Optional[str]=None) -> pd.DataFrame:
    # create output path if not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    needed_info = ['optimizer', 'lr_scheduller', 'learning_rate_max']
    runs = get_top_n_sweep_runs(sweep_id, top_n, metric)
    info = get_config_info_about_runs(runs, needed_info)
    metainfo = pd.DataFrame(columns=["ID"]+needed_info)
    for run in runs:
        download_model_from_run(run, output_path, model_name)
        # model name template: language_windowLength_sweepID  (e.g. english_80_99)
        new_model_name = new_model_name_prefix if new_model_name_prefix else ""
        new_model_name += "%s"%run.name.split('-')[-1]
        os.rename(os.path.join(output_path, model_name), os.path.join(output_path, new_model_name))
        print("Model %s downloaded from run %s"%(new_model_name, run.name))
        info[run.name]['ID'] = new_model_name
        df_to_append = np.array([info[run.name]['ID'], info[run.name]['optimizer'],
                                 info[run.name]['lr_scheduller'], info[run.name]['learning_rate_max']])[np.newaxis,...]
        df_to_append = pd.DataFrame(df_to_append, columns=metainfo.columns)
        metainfo = metainfo.append(df_to_append)
    return metainfo



def main():
    language = "french"
    sweep_ids = ["denisdresvyanskiy/Engagement_recognition_fusion/2uh2sz5f", # self_attention_2Flow_all_vs_french_window_80
                 "denisdresvyanskiy/Engagement_recognition_fusion/98b53t4g", # self_attention_2Flow_all_vs_french_window_60
                 "denisdresvyanskiy/Engagement_recognition_fusion/ts34dcd4", # self_attention_2Flow_all_vs_french_window_40
                 "denisdresvyanskiy/Engagement_recognition_fusion/71snknoh"] # self_attention_2Flow_all_vs_french_window_20
    output_path = "/work/home/dsu/Model_weights/weights_of_best_models/Fusion/Cross_corpus/%s/%s"\
                  %(language.capitalize(), "self_attention_2Flow")
    metainfos = []
    for sweep_id in sweep_ids:
        sweep_info = get_sweep_info(sweep_id)
        print("processing sweep_id: %s"%sweep_info['name'])
        window_length = sweep_info["name"].split("_")[-1]
        new_model_name_prefix = "%s_%s_"%(language, window_length)
        metainfo=download_models_from_sweep(sweep_id=sweep_id, top_n=5, metric="best_val_recall",
                                            output_path=output_path, model_name="best_model.pt",
                               new_model_name_prefix=new_model_name_prefix)
        metainfo["window_length"] = window_length
        metainfo= metainfo[["ID","window_length", "optimizer", "lr_scheduller", "learning_rate_max" ]]
        metainfos.append(metainfo)

    # print all metainfos
    metainfos = pd.concat(metainfos, axis=0)
    print(metainfos)
    metainfos.to_csv(os.path.join(output_path, "info.csv"), index=False)




if __name__ == '__main__':
    main()