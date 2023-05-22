import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import selene_sdk
from selene_sdk import sequences


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


class PuffinD(nn.Module):
    def __init__(self):
        """
        Parameters
        ----------
        """
        super(PuffinD, self).__init__()
        self.uplblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, 64, kernel_size=17, padding=8), nn.BatchNorm1d(64)
                ),
                nn.Sequential(
                    nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
            ]
        )

        self.upblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(64, 64, fused=True), ConvBlock(64, 64, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(96, 96, fused=True), ConvBlock(96, 96, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
            ]
        )

        self.downlblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(128, 96, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(96, 64, kernel_size=17, padding=8),
                    nn.BatchNorm1d(64),
                ),
            ]
        )

        self.downblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(96, 96, fused=True), ConvBlock(96, 96, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(64, 64, fused=True), ConvBlock(64, 64, fused=True)
                ),
            ]
        )

        self.uplblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
            ]
        )

        self.upblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(96, 96, fused=True), ConvBlock(96, 96, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
            ]
        )

        self.downlblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(128, 96, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(96, 64, kernel_size=17, padding=8),
                    nn.BatchNorm1d(64),
                ),
            ]
        )

        self.downblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(96, 96, fused=True), ConvBlock(96, 96, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(64, 64, fused=True), ConvBlock(64, 64, fused=True)
                ),
            ]
        )
        self.final = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 10, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x, full_length_output=False):
        """Forward propagation of a batch."""
        out = x
        encodings = []
        for i, lconv, conv in zip(
            np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        encodings2 = [out]
        for enc, lconv, conv in zip(
            reversed(encodings[:-1]), self.downlblocks, self.downblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings2.append(out)

        encodings3 = [out]
        for enc, lconv, conv in zip(
            reversed(encodings2[:-1]), self.uplblocks2, self.upblocks2
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings3.append(out)

        for enc, lconv, conv in zip(
            reversed(encodings3[:-1]), self.downlblocks2, self.downblocks2
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.final(out)
        if full_length_output:
            return out
        else:
            return out


class Puffin_D:
    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda
        model_path = "./resources/puffin_D.pth"
        self.bignet = PuffinD()
        if self.use_cuda:
            self.bignet.load_state_dict(torch.load(model_path))
        else:
            self.bignet.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu")), strict=False
            )

        self.bignet.eval()
        if self.use_cuda:
            self.bignet.cuda()
        else:
            self.bignet.cpu()

    def predict(self, seq):
        seq = sequences.sequence_to_encoding(
            seq,
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

        if self.use_cuda:
            seqt = torch.FloatTensor(seq[None, :, :]).transpose(1, 2).cuda()
            with torch.no_grad():
                pred = self.bignet(seqt)
                pred = pred.cpu().numpy()
        else:
            seqt = torch.FloatTensor(seq[None, :, :]).transpose(1, 2)
            with torch.no_grad():
                pred = self.bignet(seqt)
                pred = pred.numpy()
        return pred


if __name__ == "__main__":
    import sys
    from docopt import docopt

    doc = """
    Puffin outputs transcription inititation signal prediciton for the input genome sequence.
    
    Usage:
    puffin_D_predict coord [options] <coordinate>
    puffin_D_predict sequence [options] <fasta_file_path>
    puffin_D_predict region [options] <tsv_file>
    
    Options:
    -h --help        Show this screen.
    --use_cuda    Use CUDA to make the prediciton
    """

    if len(sys.argv) == 1:
        sys.argv.append("-h")

    arguments = docopt(doc)

    genome_path = "./resources/hg38.fa"
    genome = selene_sdk.sequences.Genome(genome_path)
    if arguments["--use_cuda"]:
        use_cuda = True
    else:
        use_cuda = False

    puffin_d = Puffin_D(use_cuda)

    seq_list = []
    name_list = []
    if arguments["coord"]:
        chrm, poss = arguments["<coordinate>"].split(":")
        strand = arguments["<coordinate>"][-1]
        start = int(poss[:-1]) - 50000
        end = int(poss[:-1]) + 50000
        print(start, end)

        if strand == "-":
            offset = 1
            strand_ = "minus"
        else:
            offset = 0
            strand_ = "plus"

        seq = genome.get_sequence_from_coords(
            chrm, start + offset, end + offset, strand
        )
        seq_list.append(seq)
        name = "puffin_D_" + chrm + "_" + str(start) + "_" + str(end) + "_" + strand_
        name_list.append(name)

    if arguments["sequence"]:
        fasta_file = open(arguments["<fasta_file_path>"], "r")
        for line in fasta_file:
            if line[0] == ">":
                name = line[1:-1]
            else:
                seq = line

                if len(seq) != 100000:
                    print(
                        "The input sequence lenght should be 100Kbp, current sequence length is "
                        + str(len(seq))
                        + " bps"
                    )
                    continue
                else:
                    seq_list.append(seq)

                    for s in [
                        "#", "%", "&", "{", "}", "<", ">", "*", "?",
                        "/", " ", "$", "!", "'", '"', ":", "@", "+", "`", "|", "=",
                    ]:
                        name = name.replace(s, "_")
                    name_list.append(name)

    if arguments["region"]:
        tsv_file = pd.read_csv(arguments["<tsv_file>"], sep="\t")

        for i in range(len(tsv_file)):
            chrm = tsv_file["chr"].values[i]
            start = tsv_file["start"].values[i]
            end = tsv_file["end"].values[i]
            strand = tsv_file["strand"].values[i]
            start = int(start)
            end = int(end)

            if strand == "-":
                offset = 1
                strand_ = "minus"
            else:
                offset = 0
                strand_ = "plus"

            seq = genome.get_sequence_from_coords(
                chrm, start + offset, end + offset, strand
            )

            if len(seq) != 100000:
                print(
                    "The input sequence lenght should be 100Kbp, current sequence length is "
                    + str(len(seq))
                    + " bps"
                )
                continue
            else:
                seq_list.append(seq)
                name = (
                    "puffin_D_"
                    + chrm
                    + "_"
                    + str(start)
                    + "_"
                    + str(end)
                    + "_"
                    + strand_
                )
                name_list.append(name)

    for seq, name in zip(seq_list, name_list):
        pred = puffin_d.predict(seq)
        np.save(name, pred)
