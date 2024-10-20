from lightning.pytorch.callbacks import ModelCheckpoint
from finetune_model import PuffinFinetuned
from selene_sdk import sequences
import torch
import pandas as pd
import argparse

def main(sequence, filename):

    model = PuffinFinetuned.load_from_checkpoint("model_all_new.ckpt")
    
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
    seq = torch.Tensor(seq).view(-1, 2000, 4).cuda()
    y_pred_plus, y_pred_minus = model(seq)
    y_pred_plus = y_pred_plus[:,325:-325].cpu().tolist()[0]
    y_pred_minus = y_pred_minus[:,325:-325].cpu().tolist()[0]
    sequence = list(sequence)[325:-325]
    pd.DataFrame([sequence, y_pred_plus, y_pred_minus], index=["Nucleotide", "Pred_Plus", "Pred_Minus"]).to_csv(filename, header=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-s', '--sequence', required=True)
    parser.add_argument('-f', '--filename', default="puffin_finetuned_results.csv")
    
#    sequence = "ATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACAATATACATATAGATAGATACA"
    filename = "puffin_finetuned_results.csv"
    main(args.sequence, args.filename)