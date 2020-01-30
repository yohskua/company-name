import tensorflow as tf

# enter graph mode, usually faster
tf.compat.v1.disable_eager_execution()
from rnn_vae import RnnVae
import pandas as pd

df = pd.read_csv("~/datasets/companies_sorted.csv", sep=",")
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
df_tmp = df_tmp[df_tmp.len_name <= 25]

X = df_tmp.name.to_list()
X_dummy = ["google"] * 100
rnn_vae = RnnVae(X, GO="A", EOS="Z", latent_dim=128, intermediate_dim=256)
rnn_vae_dummy = RnnVae(X_dummy, GO="A", EOS="Z", latent_dim=128, intermediate_dim=256)
rnn_vae.train(epochs=1, batch_size=16)
rnn_vae_dummy.train(epochs=30, batch_size=16)

# dummy demo
word = "google"
for _ in range(1000):
    sampled_around = rnn_vae_dummy.sample_around(word)
    if sampled_around != word:
        print(sampled_around)

# normal demo
# problem is now we always sample around the same data point distribution: 0 mean and 1 std.
for _ in range(30):
    print(rnn_vae.sample_prediction())
