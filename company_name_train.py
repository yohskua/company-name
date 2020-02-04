import os
import argparse

import datetime
import pandas as pd
import tensorflow as tf

# enter graph mode, usually faster
tf.compat.v1.disable_eager_execution()

from rnn_vae import RnnVae


def select_names():
    df = pd.read_csv("~/datasets/companies_sorted.csv", sep=",")
    df = df[df.industry.isin(["information technology and services", "internet"])]
    df = df[df["current employee estimate"] > 5]
    df.dropna(subset=["name"], inplace=True)

    def allow_name(name):
        for token in name:
            try:
                token.encode("ascii")
            except:
                return False
        return True

    # remove chinese names and others that use characters we do not want to use
    df["name_allowed"] = [allow_name(name) for name in df.name]
    df = df[df["name_allowed"] == True]

    df["len_name"] = [len(name) for name in df.name]
    # remove too long names, hard to encode and we are not interested in them
    df = df[df.len_name <= 12]
    return df.name.to_list()


def current_date():
    current_date = datetime.datetime.now()
    return current_date.strftime("%Y-%m-%d-%H%M")


def make_dir(output_path):
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    except Exception as e:
        raise e


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epochs", default=1, type=int, help="")
    argparser.add_argument("--batch-size", default=16, type=int, help="")
    argparser.add_argument("--latent-dim", default=128, type=int, help="")
    argparser.add_argument("--intermediate-dim", default=256, type=int, help="")
    argparser.add_argument(
        "--save-pred-every",
        default=5,
        type=int,
        help="save model predictions every x epochs",
    )
    argparser.add_argument(
        "--predictions", default=100, type=int, help="number of predictions to make"
    )

    args = argparser.parse_args()
    df_preds = pd.DataFrame()
    current_date = current_date()
    output_path = os.path.join(os.path.expanduser("~"), "tmp", current_date)
    make_dir(output_path)
    names = select_names()
    rnn_vae = RnnVae(
        names,
        GO="A",
        EOS="Z",
        latent_dim=args.latent_dim,
        intermediate_dim=args.intermediate_dim,
    )
    for i in range(args.epochs):
        rnn_vae.train(epochs=1, batch_size=args.batch_size)
        if (i + 1) % args.save_pred_every == 0:
            predictions = [rnn_vae.sample_prediction() for _ in range(args.predictions)]
            df_preds["epoch_{}".format(i + 1)] = [
                pred for pred in predictions if pred in set(names)
            ]

            # save early to check predictions early from disk. files should not be big
            df_preds.to_csv(os.path.join(output_path, "predictions.csv"), index=False)
