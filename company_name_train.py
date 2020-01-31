import os
import argparse

import datetime
import tensorflow as tf

# enter graph mode, usually faster
tf.compat.v1.disable_eager_execution()

from rnn_vae import RnnVae
import pandas as pd

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epochs", default=1, type=int, help="")
    argparser.add_argument("--batch-size", default=16, type=int, help="")
    argparser.add_argument("--latent-dim", default=128, type=int, help="")
    argparser.add_argument("--intermediate-dim", default=256, type=int, help="")
    args = argparser.parse_args()

    df = pd.read_csv("~/datasets/companies_sorted.csv", sep=",", nrows=100)
    df_tmp = df[df.industry.isin(["information technology and services", "internet"])]
    df_tmp = df_tmp[df_tmp["current employee estimate"] > 5]
    df_tmp.dropna(subset=["name"], inplace=True)

    def allow_name(name):
        for token in name:
            try:
                token.encode("ascii")
            except:
                return False
        return True

    # remove chinese names and others that use characters we do not want to use
    df_tmp["name_allowed"] = [allow_name(name) for name in df_tmp.name]
    df_tmp = df_tmp[df_tmp["name_allowed"] == True]

    df_tmp["len_name"] = [len(name) for name in df_tmp.name]
    # remove too long names, hard to encode and we are not interested in them
    df_tmp = df_tmp[df_tmp.len_name <= 12]

    X = df_tmp.name.to_list()
    rnn_vae = RnnVae(
        X,
        GO="A",
        EOS="Z",
        latent_dim=args.latent_dim,
        intermediate_dim=args.intermediate_dim,
    )
    df_preds = pd.DataFrame()
    frequence_saving = 2
    number_predictions_epochs = 100
    current_date = datetime.datetime.now()
    current_date = current_date.strftime("%Y-%m-%d-%H%M")
    output_path = os.path.join(os.path.expanduser("~"), "tmp", current_date)
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    except Exception as e:
        print(e)
        raise (e)

    for i in range(10):
        rnn_vae.train(epochs=1, batch_size=args.batch_size)
        if (i + 1) % frequence_saving == 0:
            df_preds["epoch_{}".format(i + 1)] = [
                rnn_vae.sample_prediction() for _ in range(number_predictions_epochs)
            ]
            # allows to check predictions early from disk. files should not be big
            df_preds.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

