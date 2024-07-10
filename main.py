import os, sys
from argparse import Namespace
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import combined_loader
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.preprocessing import LabelEncoder
import paho.mqtt.client as mqtt
from csv_mqtt.csv_mqtt import CsvMqtt
import csv
import time

os.environ["CUDA_VISIBLE_DEVICES"]=""
datafile_path = Path('PyTorchAnomalyDD.csv')
datasets_root = Path('working')


raw_dt = pd.read_csv(datafile_path, nrows = 28800)
raw_dt = raw_dt[raw_dt['Occupancy Mode Indicator'] > 0]
raw_dt.head()
raw_dt.describe()

data = raw_dt.copy()
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.plot(x='Datetime', y='AHU: Outdoor Air Temperature', figsize=(12, 8))
plt.ylabel("Temperature")
plt.title("Original Dataset")

data['day'] = data['Datetime'].dt.day
data['month'] = data['Datetime'].dt.month
data['hour_min'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute / 60
data['day_of_week'] = data['Datetime'].dt.dayofweek
data['t'] = (data['Datetime'].astype(np.int64)/1e11).astype(np.int64)
data.drop('Datetime', axis=1, inplace=True)

cont_vars = ['AHU: Outdoor Air Temperature', 'hour_min', 't']
cat_vars = ['day', 'month', 'day_of_week']

label_encoders = [LabelEncoder() for _ in cat_vars]
for col, enc in zip(cat_vars, label_encoders):
    data[col] = enc.fit_transform(data[col])

test_ratio = 0.3
tr_data = data.iloc[: int(len(data) * (1 - test_ratio))]
tst_data = data.iloc[int(len(data) * (1 - test_ratio)):]
scaler = preprocessing.StandardScaler().fit(tr_data[cont_vars])
tr_data_scaled = tr_data.copy()
tr_data_scaled[cont_vars] = scaler.transform(tr_data[cont_vars])
tst_data_scaled = tst_data.copy()
tst_data_scaled[cont_vars] = scaler.transform(tst_data[cont_vars])

tr_data_scaled.to_csv(datasets_root/'train.csv', index=False)
tst_data_scaled.to_csv(datasets_root/'test.csv', index=False)


class TSDataset(Dataset):
    def __init__(self, split, cont_vars=None, cat_vars=None, lbl_as_feat=True):
        super().__init__()
        assert split in ['train', 'test', 'both']
        self.lbl_as_feat = lbl_as_feat
        if split == 'train':
            self.df = pd.read_csv(datasets_root / 'train.csv')
        elif split == 'test':
            self.df = pd.read_csv(datasets_root / 'test.csv')
        else:
            df1 = pd.read_csv(datasets_root / 'train.csv')
            df2 = pd.read_csv(datasets_root / 'test.csv')
            self.df = pd.concat((df1, df2), ignore_index=True)
        if cont_vars:
            self.cont_vars = cont_vars
            if self.lbl_as_feat:
                try:
                    assert 'AHU: Outdoor Air Temperature' in self.cont_vars
                except AssertionError:
                    self.cont_vars.insert(0, 'AHU: Outdoor Air Temperature')
            else:
                try:
                    assert 'AHU: Outdoor Air Temperature' not in self.cont_vars
                except AssertionError:
                    self.cont_vars.remove('AHU: Outdoor Air Temperature')

        else:
            self.cont_vars = ['AHU: Outdoor Air Temperature', 'hour_min', 't']

        if cat_vars:
            self.cat_vars = cat_vars
        else:
            self.cat_vars = ['day', 'month', 'day_of_week']

        if self.lbl_as_feat:
            self.cont = self.df[self.cont_vars].copy().to_numpy(dtype=np.float32)
        else:
            self.cont = self.df[self.cont_vars].copy().to_numpy(dtype=np.float32)
            self.lbl = self.df['AHU: Outdoor Air Temperature'].copy().to_numpy(dtype=np.float32)
        self.cat = self.df[self.cat_vars].copy().to_numpy(dtype=np.int64)

    def __getitem__(self, idx):
        if self.lbl_as_feat:
            return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx])
        else:
            return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx]), torch.tensor(self.lbl[idx])

    def __len__(self):
        return self.df.shape[0]

ds = TSDataset(split='both', cont_vars=['AHU: Outdoor Air Temperature', 't'], cat_vars=['day_of_week'], lbl_as_feat=True)
# print(len(ds))
it = iter(ds)
# for _ in range(10):
    # print(next(it))


class Layer(nn.Module):

    def __init__(self, in_dim, out_dim, bn=True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if bn: layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):

    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)
        self.embeds = nn.ModuleList([
            nn.Embedding(n_cats, emb_size) for (n_cats, emb_size) in self.hparams.embedding_sizes
        ])
        in_dim = sum(emb.embedding_dim for emb in self.embeds) + len(self.hparams.cont_vars)
        layer_dims = [in_dim] + [int(s) for s in self.hparams.layer_sizes.split(',')]
        bn = self.hparams.batch_norm
        self.layers = nn.Sequential(
            *[Layer(layer_dims[i], layer_dims[i + 1], bn) for i in range(len(layer_dims) - 1)],
        )
        self.mu = nn.Linear(layer_dims[-1], self.hparams.latent_dim)
        self.logvar = nn.Linear(layer_dims[-1], self.hparams.latent_dim)

    def forward(self, x_cont, x_cat):
        x_embed = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x_embed = torch.cat(x_embed, dim=1)
        x = torch.cat((x_embed, x_cont), dim=1)
        h = self.layers(x)
        mu_ = self.mu(h)
        logvar_ = self.logvar(h)
        return mu_, logvar_, x


class Decoder(nn.Module):
    def __init__(self, **hparams):
        super().__init__()
        self.hparams = Namespace(**hparams)
        hidden_dims = [self.hparams.latent_dim] + [int(s) for s in reversed(self.hparams.layer_sizes.split(','))]
        out_dim = sum(emb_size for _, emb_size in self.hparams.embedding_sizes) + len(self.hparams.cont_vars)
        bn = self.hparams.batch_norm
        self.layers = nn.Sequential(
            *[Layer(hidden_dims[i], hidden_dims[i + 1], bn) for i in range(len(hidden_dims) - 1)],
        )
        self.reconstructed = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, z):
        h = self.layers(z)
        recon = self.reconstructed(h)
        return recon


class VAE(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)

    def reparameterize(self, mu, logvar):

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * self.hparams.stdev
            return eps * std + mu
        else:
            return mu

    def forward(self, batch):
        x_cont, x_cat = batch
        assert x_cat.dtype == torch.int64
        mu, logvar, x = self.encoder(x_cont, x_cat)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, x

    def loss_function(self, obs, recon, mu, logvar):
        recon_loss = F.smooth_l1_loss(recon, obs, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        return recon_loss, kld

    def training_step(self, batch, batch_idx):
        recon, mu, logvar, x = self.forward(batch)
        recon_loss, kld = self.loss_function(x, recon, mu, logvar)
        loss = recon_loss + self.hparams.kld_beta * kld

        self.log('total_loss', loss.mean(dim=0), on_step=True, prog_bar=True,
                 logger=True)
        self.log('recon_loss', recon_loss.mean(dim=0), on_step=True, prog_bar=True,
                 logger=True)
        self.log('kld', kld.mean(dim=0), on_step=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        recon, mu, logvar, x = self.forward(batch)
        recon_loss, kld = self.loss_function(x, recon, mu, logvar)
        loss = recon_loss + self.hparams.kld_beta * kld
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay,
                                eps=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=25, T_mult=1, eta_min=1e-9, last_epoch=-1)
        return [opt], [sch]

    def train_dataloader(self):
        dataset = TSDataset('train', cont_vars=self.hparams.cont_vars,
                            cat_vars=self.hparams.cat_vars, lbl_as_feat=True
                            )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=2,
                          pin_memory=True, persistent_workers=True, shuffle=True
                          )

    def test_dataloader(self):
        dataset = TSDataset('test', cont_vars=self.hparams.cont_vars,
                            cat_vars=self.hparams.cat_vars, lbl_as_feat=True
                            )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=2,
                          pin_memory=True, persistent_workers=True
                          )

cont_features = ['AHU: Outdoor Air Temperature', 'hour_min', 't']
cat_features = ['day_of_week']

embed_cats = [len(tr_data_scaled[c].unique()) for c in cat_features]

hparams = OrderedDict(
    run='embsz16_latsz16_bsz128_lay64-128-256-128-64_ep100_cosineWR_v1',
    cont_vars = cont_features,
    cat_vars = cat_features,
    embedding_sizes = [(embed_cats[i], 16) for i in range(len(embed_cats))],
    latent_dim = 16,
    layer_sizes = '64,128,256,128,64',
    batch_norm = True,
    stdev = 0.1,
    kld_beta = 0.05,
    lr = 0.001,
    weight_decay = 1e-5,
    batch_size = 128,
    epochs = 60,
)

model = VAE(**hparams)

ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath='.', filename='vae_weights')
trainer = pl.Trainer(accelerator='auto', devices=1, strategy='auto',
                     max_epochs=hparams['epochs'], benchmark=True,
                     callbacks=[ckpt_callback], gradient_clip_val=10., enable_model_summary=True,
)


def on_connect(client, userdata, flags, rc_value):
    """
    On connect callback handler for MQTT.
    """
    print(f"Connected with result code : {rc_value}")

def on_publish(client, userdata, mid):
    """
    On Publish callback handler for MQTT.
    """
    print("Message Published.")

class CsvMqtt:
    """
    Class providing the methods for CSV-MQTT Connectors.
    """

    def __init__(self, broker, port=1883, timeout=60,
                 connect_cb=on_connect, publish_cb=on_publish):
        self.mqtt_client = mqtt.Client(clean_session=1)
        self.mqtt_client.on_connect = connect_cb
        self.mqtt_client.on_publish = publish_cb
        self.mqtt_client.connect(broker, port, timeout)

    def publish_csv_data(self, mqtt_topic, csv_path="PyTorchAnomalyDD.csv", interval=1):
        """
        The method to publish the contents of the csv in given
        interval.
        """
        if not os.path.exists(csv_path):
            return False
        data_value = {}
        key_values = []

        with open(csv_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for (row, row2) in zip(raw_dt['AHU: Outdoor Air Temperature'], raw_dt['Datetime']):
                temp_data_val = "Time: " + row2 + "\nTemperature: " + str(row)
                try:
                    self.mqtt_client.publish(mqtt_topic, temp_data_val)
                    time.sleep(interval)
                except Exception as excep_v:
                    print(f"Publish Failed. {excep_v}")
        return True

def main():
    connector = CsvMqtt("broker.emqx.io")
    connector.publish_csv_data("test")
    # trainer.fit(model)
    # trainer.test(model)
    dataset = TSDataset('test', cont_vars=hparams['cont_vars'],
                        cat_vars=['day_of_week'],
                        lbl_as_feat=True)

    trained_model = VAE.load_from_checkpoint('./vae_weights-v1.ckpt')
    trained_model.freeze()


    losses = []
    for i in range(len(dataset)):
        x_cont, x_cat = dataset[i]
        x_cont.unsqueeze_(0)
        x_cat.unsqueeze_(0)
        recon, mu, logvar, x = trained_model.forward((x_cont, x_cat))
        recon_loss, kld = trained_model.loss_function(x, recon, mu, logvar)
        losses.append(recon_loss + trained_model.hparams.kld_beta * kld)
    data_with_losses = dataset.df
    data_with_losses['loss'] = np.asarray(losses)
    data_with_losses.sort_values('t', inplace=True)

    quant = 0.99
    thresh = data_with_losses['loss'].quantile(quant)
    #print(thresh)
    data_with_losses['anomaly'] = data_with_losses['loss'] > thresh

    data_with_losses_unscaled = data_with_losses.copy()
    data_with_losses_unscaled[cont_vars] = scaler.inverse_transform(data_with_losses[cont_vars])
    for enc, var in zip(label_encoders, cat_vars):
        data_with_losses_unscaled[var] = enc.inverse_transform(data_with_losses[var])
    data_with_losses_unscaled = pd.DataFrame(data_with_losses_unscaled, columns=data_with_losses.columns)
    data_with_losses_unscaled['Datetime'] = pd.to_datetime(data_with_losses_unscaled['t'] * 1e11, unit='ns')

    if len(data_with_losses['anomaly']) > 0:
        anomalies_ts = data_with_losses_unscaled.loc[
            data_with_losses_unscaled['anomaly'], ('Datetime', 'AHU: Outdoor Air Temperature')]

        print(f"Anomaly detected at times: ")
        print(anomalies_ts['Datetime'])


    print(data_with_losses_unscaled.head())
    anomalies_ts = data_with_losses_unscaled.loc[data_with_losses_unscaled['anomaly'], ('Datetime', 'AHU: Outdoor Air Temperature')]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data_with_losses_unscaled['Datetime'], data_with_losses_unscaled['AHU: Outdoor Air Temperature'], alpha=.5)
    ax.scatter(anomalies_ts['Datetime'], anomalies_ts['AHU: Outdoor Air Temperature'], color='red', label='anomaly')
    plt.legend()
    plt.xlabel("Datetime")
    plt.ylabel("temperature")
    plt.title("Location of the Anomalies on the Time Series, Test Period")
    plt.show()

if __name__ == '__main__':
    main()
