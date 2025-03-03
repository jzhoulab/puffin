import pandas as pd
import selene_sdk
from tqdm import tqdm
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary
from finetune_model import PuffinFinetuned, PandasDataModule
import pyBigWig
import pyranges as pr
import time
tqdm.pandas()
generate_bigwig = False

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

    logger = WandbLogger(project=f"puffing finetuning_test", log_model="all", name=f"Test: {test_chr}, Val: {val_chr}")
    model_folder = f"models/puffin_finetuned/"

    genomic_data_module = PandasDataModule(promoters, val_chr=[val_chr], test_chr=[test_chr], batch_size=256)

    model = PuffinFinetuned.load_from_checkpoint("model_all_new.ckpt")
    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        dirpath=model_folder,
        filename=f"best_val_loss_scatacpredictor",
        mode="min"
    )

    trainer = pl.Trainer(logger=logger, callbacks=[ModelSummary(max_depth=3), checkpoint_callback_best], max_epochs=500, num_sanity_val_steps=1) # add gradient_clip_val=1 if needed  , accumulate_grad_batches=2 <- same with this

    logger.watch(model, log="all", log_freq=10)

    trainer.test(model, datamodule=genomic_data_module)
    preds = trainer.predict(model, datamodule=genomic_data_module)

    if(generate_bigwig):
        chromsizes = pd.read_csv("hg38.chrom.sizes", sep="\t", index_col=0, header=None, names=["start"]).to_dict()["start"]
        all_values_plus = []
        all_values_minus = []
        puffin_predictions_names = ["FANTOM_CAGE_plus", "ENCODE_CAGE_plus", "ENCODE_RAMPAGE_plus", "GRO_CAP_plus", "PRO_CAP_plus",
                                    "FANTOM_CAGE_minus", "ENCODE_CAGE_minus", "ENCODE_RAMPAGE_minus", "GRO_CAP_minus", "PRO_CAP_minus"]
        puffin_predictions = []
        for i in range(0, 10):
            puffin_predictions.append([])
        for pred in preds:
            for i in range(0, pred[0].shape[0]):
                y_real_plus = pred[0][i]
                y_real_minus = pred[1][i]
                y_pred_plus = pred[2][i].tolist()
                y_pred_minus = pred[3][i].tolist()
                chromosome = pred[5][i]
                start = int(pred[6][i])
                end = int(pred[7][i])
                for j in range(0, len(y_real_plus)):
                    all_values_plus.append([chromosome, start+j, start+j+1, y_pred_plus[j], "+"])
                    all_values_minus.append([chromosome, start+j, start+j+1, y_pred_minus[j], "-"])
                    for k in range(0, 10):
                        y_pred_puffin = pred[4][i][k].tolist()
                        puffin_predictions[k].append([chromosome, start+j, start+j+1, y_pred_puffin[j], "+"])
                pass
        gr_df = pd.DataFrame(all_values_plus, columns=["Chromosome", "Start", "End", "Value", "Strand"])
        gr_df = gr_df.groupby(["Chromosome", "Start", "End", "Strand"]).mean().reset_index() # mean of duplicate predictions
        gr_df = pr.PyRanges(gr_df)
        gr_df.to_bigwig("results/results_all_plus.bw", chromsizes, rpm=False, value_col="Value")

        gr_df = pd.DataFrame(all_values_minus, columns=["Chromosome", "Start", "End", "Value", "Strand"])
        gr_df = gr_df.groupby(["Chromosome", "Start", "End", "Strand"]).mean().reset_index() # mean of duplicate predictions
        gr_df = pr.PyRanges(gr_df)
        gr_df.to_bigwig("results/results_all_minus.bw", chromsizes, rpm=False, value_col="Value")

        for i, name in enumerate(puffin_predictions_names):
            gr_df = pd.DataFrame(puffin_predictions[i], columns=["Chromosome", "Start", "End", "Value", "Strand"])
            gr_df = gr_df.groupby(["Chromosome", "Start", "End", "Strand"]).mean().reset_index() # mean of duplicate predictions
            gr_df = pr.PyRanges(gr_df)
            gr_df.to_bigwig(f"results/puffin_{name}.bw", chromsizes, rpm=False, value_col="Value")

        pass
