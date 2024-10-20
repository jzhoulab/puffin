import pandas as pd
import selene_sdk
from selene_sdk import sequences
from tqdm import tqdm
import pyBigWig
from torch.utils.data import Dataset, DataLoader
import torch
import lightning.pytorch as pl
import numpy as np
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, R2Score, PearsonCorrCoef, SpearmanCorrCoef, ConcordanceCorrCoef, RelativeSquaredError
from torchmetrics import MetricCollection
from puffin import Puffin
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import shutil
from lightning.pytorch.callbacks import ModelSummary
import time
import torch.nn as nn
tqdm.pandas()

normal_chromosomes = ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8", "chr9", "chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"]
num_workers_loader = 16

class PuffinFinetuned(pl.LightningModule):
    def __init__(self, lr=0.0001):
        super().__init__()
        self.example_input_array = (torch.Tensor(8, 2000, 4)) 
        self.lr = lr
        self.save_hyperparameters()

        self.puffin = Puffin()
        self.puffin_clean = Puffin()
        for param in self.puffin_clean.parameters():
            param.requires_grad = False
        self.last_layer_plus = nn.Conv1d(10, 1, 1)
        self.last_layer_minus = nn.Conv1d(10, 1, 1)

        metrics = MetricCollection([MeanAbsoluteError(), MeanAbsolutePercentageError(), MeanSquaredError(), R2Score(), PearsonCorrCoef(), SpearmanCorrCoef(), ConcordanceCorrCoef(), RelativeSquaredError()])
        
        self.puffin_predictions_names = ["FANTOM_CAGE_plus", "ENCODE_CAGE_plus", "ENCODE_RAMPAGE_plus", "GRO_CAP_plus", "PRO_CAP_plus",
                                "FANTOM_CAGE_minus", "ENCODE_CAGE_minus", "ENCODE_RAMPAGE_minus", "GRO_CAP_minus", "PRO_CAP_minus"]
        self.puffin_metrics = {}
        for name in self.puffin_predictions_names:
            self.puffin_metrics[name] = metrics.clone(prefix=f"{name}_")
        self.train_metrics_plus = metrics.clone(prefix='train_plus_')
        self.valid_metrics_plus = metrics.clone(prefix='val_plus_')
        self.train_metrics_minus = metrics.clone(prefix='train_minus_')
        self.valid_metrics_minus = metrics.clone(prefix='val_minus_')
        self.pearson_table_plus = []
        self.pearson_table_minus = []
                
    def forward(self, seq):
        batch_size = seq.shape[0]
        seq = seq.transpose(1, 2)
        puffin_result = self.puffin(seq)
        return self.last_layer_plus(puffin_result).view(batch_size, -1), self.last_layer_minus(puffin_result).view(batch_size, -1)


    def process_batch(self, batch):
        seq, y_plus, y_minus, chromosome, start, end, strand = batch # each batch has sequence, rnaseq data, y -> I assume scATAC-Seq data, and position (just in case - for debugging  etc.)
        
        y_pred_plus, y_pred_minus = self(seq)
        y_plus = y_plus[:,325:-325]
        y_minus = y_minus[:,325:-325]
        y_pred_plus = y_pred_plus[:,325:-325]
        y_pred_minus = y_pred_minus[:,325:-325]
        loss = torch.nn.MSELoss(reduction='none')
        loss = loss(torch.concat([y_pred_plus, y_pred_minus]), torch.concat([y_plus, y_minus])).mean()

        return loss, seq, y_pred_plus, y_pred_minus, y_plus, y_minus, chromosome, start+325, end-325, strand

    def training_step(self, batch, batch_idx):
        loss, seq, y_pred_plus, y_pred_minus, y_plus, y_minus, chromosome, start, end, strand = self.process_batch(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=seq.shape[0], sync_dist=True)
        self.log_dict(self.train_metrics_plus(y_pred_plus.reshape(-1), y_plus.reshape(-1)), sync_dist=True, on_epoch=True, batch_size=seq.shape[0])
        self.log_dict(self.train_metrics_minus(y_pred_minus.reshape(-1), y_minus.reshape(-1)), sync_dist=True, on_epoch=True, batch_size=seq.shape[0]) 
        return loss
    
    def on_train_epoch_end(self):
        self.train_metrics.reset()
        print('\n')

    def validation_step(self, batch, batch_idx):
        loss, seq, y_pred_plus, y_pred_minus, y_plus, y_minus, chromosome, start, end, strand = self.process_batch(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=seq.shape[0], sync_dist=True)
        
        self.log_dict(self.valid_metrics_plus(y_pred_plus.reshape(-1), y_plus.reshape(-1)), sync_dist=True, on_epoch=True, batch_size=seq.shape[0])
        self.log_dict(self.valid_metrics_minus(y_pred_minus.reshape(-1), y_minus.reshape(-1)), sync_dist=True, on_epoch=True, batch_size=seq.shape[0])

    def on_validation_epoch_end(self):
        self.valid_metrics.reset()
        
    def test_step(self, batch, batch_idx):
        loss, seq, y_pred_plus, y_pred_minus, y_plus, y_minus, chromosome, start, end, strand = self.process_batch(batch)
        puffin_predictions = self.puffin_clean(seq.transpose(1, 2))[:,:,325:-325]
        self.log_dict(self.valid_metrics_plus(y_pred_plus.reshape(-1), y_plus.reshape(-1)), sync_dist=True, on_epoch=True, batch_size=seq.shape[0])
        self.log_dict(self.valid_metrics_minus(y_pred_minus.reshape(-1), y_minus.reshape(-1)), sync_dist=True, on_epoch=True, batch_size=seq.shape[0])

        for i in range(0, seq.shape[0]):
            pearson = PearsonCorrCoef().to(y_plus.device)
            pearson_calculated = pearson(y_pred_plus[i].view(-1), y_plus[i].view(-1))
            self.pearson_table_plus.append([chromosome[i], start[i].item(), end[i].item(), strand[i], pearson_calculated.item()])
        for i in range(0, seq.shape[0]):
            pearson = PearsonCorrCoef().to(y_minus.device)
            pearson_calculated = pearson(y_pred_minus[i].view(-1), y_minus[i].view(-1))
            self.pearson_table_minus.append([chromosome[i], start[i].item(), end[i].item(), strand[i], pearson_calculated.item()])
        
        for i, name in enumerate(self.puffin_predictions_names):
            if("_plus" in name):
                self.log_dict(self.puffin_metrics[name](puffin_predictions[:,i,:].reshape(-1).cpu(), y_plus.reshape(-1).cpu()), sync_dist=True, on_epoch=True, batch_size=seq.shape[0])
            else:
                self.log_dict(self.puffin_metrics[name](puffin_predictions[:,i,:].reshape(-1).cpu(), y_minus.reshape(-1).cpu()), sync_dist=True, on_epoch=True, batch_size=seq.shape[0])

        return y_plus, y_minus,y_pred_plus, y_pred_minus, puffin_predictions, chromosome, start, end
    
    def predict_step(self, batch, batch_idx):
        loss, seq, y_pred_plus, y_pred_minus, y_plus, y_minus, chromosome, start, end, strand = self.process_batch(batch)
        puffin_predictions = self.puffin_clean(seq.transpose(1, 2))[:,:,325:-325]

        return y_plus, y_minus,y_pred_plus, y_pred_minus, puffin_predictions, chromosome, start, end
    
    def on_test_epoch_end(self):
        self.logger.log_table(key="pearson_plus", columns=["chr", "pos", "end", "strand", "pearson"], data=self.pearson_table_plus)
        self.logger.log_table(key="pearson_minus", columns=["chr", "pos", "end", "strand", "pearson"], data=self.pearson_table_minus)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class PandasDataset(Dataset):
    def __init__(self, dataframe, chromosomes=None):
        self.df = dataframe[dataframe["chr"].isin(chromosomes)].reset_index(drop=True)
        signal_plus = pyBigWig.open("pooled.plus.bw")
        signal_minus = pyBigWig.open("pooled.minus.bw")
        self.signal_plus = {}
        self.signal_minus = {}
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            pass
            self.signal_plus[(row["chr"], row["Start"], row["End"])] = np.nan_to_num(signal_plus.values(row["chr"], row["Start"], row["End"], numpy=True), nan=0.0)
            self.signal_minus[(row["chr"], row["Start"], row["End"])] = np.nan_to_num(signal_minus.values(row["chr"], row["Start"], row["End"], numpy=True), nan=0.0)
        pass
        
    def __len__(self):
        return len(self.df)
    def get_signal(self, row):
        return self.signal_plus[(row["chr"], row["Start"], row["End"])], self.signal_minus[(row["chr"], row["Start"], row["End"])]
        
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        sequence = item["Sequence"]
        seq = sequences.sequence_to_encoding(sequence,
            base_to_index={
                "A": 0,
                "a": 0,
                "C": 1,
                "c": 1,
                "G": 2,
                "g": 2,
                "T": 3,
                "t": 3,
            },
            bases_arr="ACGT",
        )
        signal_plus, signal_minus = self.get_signal(item)
        return seq, signal_plus, signal_minus, item["chr"], item["Start"], item["End"], item["strand"]

class PandasDataModule(pl.LightningDataModule):
    def __init__(self, df, batch_size: int = 4, val_chr = ["chr9"], test_chr = ["chr8"]):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.val_chr = val_chr
        self.test_chr = test_chr

    def setup(self, stage=None):
        train_chrs = [x for x in normal_chromosomes if x not in self.val_chr+self.test_chr]
        if(len(train_chrs) > 0):
            self.genomic_train = PandasDataset(self.df, train_chrs)
        if(len(self.val_chr) > 0):
            self.genomic_val = PandasDataset(self.df, self.val_chr)
        if(len(self.test_chr) > 0):
            self.genomic_test = PandasDataset(self.df, self.test_chr)

    def train_dataloader(self):
        return DataLoader(self.genomic_train, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.genomic_val, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.genomic_test, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=False, pin_memory=True)
        
    def predict_dataloader(self):
        return DataLoader(self.genomic_test, batch_size=self.batch_size, num_workers=num_workers_loader, shuffle=False, pin_memory=True)

if __name__ == "__main__":

    WINDOW_SIZE = 2000
    promoters = pd.read_csv("puffin_promoters.txt", sep="\t")
    promoters = promoters[promoters["high confidence"]].reset_index(drop=True)
    promoters["Start"] = promoters["TSS"]-WINDOW_SIZE//2
    promoters["End"] = promoters["TSS"]+WINDOW_SIZE//2
    genome_path = "./resources/hg38.fa"
    genome = selene_sdk.sequences.Genome(input_path=genome_path)
    promoters["Sequence"] = promoters.progress_apply(lambda row: genome.get_sequence_from_coords(row["chr"], row["Start"], row["End"], row["strand"]), axis=1)

    val_chr = "chr9"
    test_chr = "chr8"

    logger = WandbLogger(project=f"puffing finetuning", log_model="all", name=f"Test: {test_chr}, Val: {val_chr}")
    model_folder = f"models/puffin_finetuned/"

    genomic_data_module = PandasDataModule(promoters, val_chr=[val_chr], test_chr=[test_chr], batch_size=256)

    model = PuffinFinetuned()
    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath=model_folder,
        filename=f"best_val_loss_scatacpredictor",
        mode="min"
    )

    trainer = pl.Trainer(logger=logger, callbacks=[ModelSummary(max_depth=3), checkpoint_callback_best], max_epochs=500, num_sanity_val_steps=1) # add gradient_clip_val=1 if needed  , accumulate_grad_batches=2 <- same with this

    if(trainer.global_rank == 0):
        if os.path.exists(model_folder) and os.path.isdir(model_folder):
            shutil.rmtree(model_folder)
            time.sleep(2)
        try:
            os.mkdir(model_folder)
        except OSError:
            pass

    logger.watch(model, log="all", log_freq=10)

    trainer.fit(model, datamodule=genomic_data_module)

    pass
