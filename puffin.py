import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_fftconv import fft_conv1d, FFTConv1d
from torch.autograd import grad
import selene_sdk
from selene_sdk import sequences


class Puffin(nn.Module):
    def __init__(self, use_cuda=False):

        super(Puffin, self).__init__()
        self.conv = nn.Conv1d(4, 10, kernel_size=51, padding=25)

        self.conv_inr = nn.Conv1d(4, 10, kernel_size=15, padding=7)
        self.conv_sim = nn.Conv1d(4, 32, kernel_size=3, padding=1)

        self.activation = nn.Softplus()
        self.softplus = nn.Softplus()

        self.deconv = FFTConv1d(10 * 2, 10, kernel_size=601, padding=300)
        self.deconv_sim = FFTConv1d(64, 10, kernel_size=601, padding=300)
        self.deconv_inr = nn.ConvTranspose1d(20, 10, kernel_size=15, padding=7)

        self.scaler = nn.Parameter(torch.ones(1))
        self.scaler2 = nn.Parameter(torch.ones(1))
        self.use_cuda = use_cuda
        self.targeti_map = {
            "FANTOM_CAGE": 0,
            "ENCODE_CAGE": 1,
            "ENCODE_RAMPAGE": 2,
            "GRO_CAP": 3,
            "PRO_CAP": 4,
        }
        self.targeti_rev_map = {
            "FANTOM_CAGE": 9,
            "ENCODE_CAGE": 8,
            "ENCODE_RAMPAGE": 7,
            "GRO_CAP": 6,
            "PRO_CAP": 5,
        }
        self.colordict = {
            "YY1+": "#1F77B4",
            "YY1-": "#c2d5e8",
            "TATA+": "#E41A1C",
            "TATA-": "#ffc6ba",
            "U1 snRNP+": "#9F9F9F",
            "U1 snRNP-": "#CFCFCF",
            "NFY+": "#00CC96",
            "NFY-": "#00cc5f",
            "ETS+": "#19d3f3",
            "ETS-": "#19e4f3",
            "SP+": "#FF7F0E",
            "SP-": "#ff930e",
            "NRF1+": "#AB63FA",
            "NRF1-": "#b663fa",
            "ZNF143+": "#17a4cf",
            "ZNF143-": "#17BECF",
            "CREB+": "#FF6692",
            "CREB-": "#ff66c2",
            "Long Inr+": "#95a2be",
            "Long Inr-": "#dde1ea",
        }

        model_path = "./resources/puffin.pth"

        self.state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.load_state_dict(self.state_dict, strict=False)

    def forward(self, x):
        y = torch.cat([self.conv(x), self.conv(x.flip([1, 2])).flip([2])], 1)
        y_sim = torch.cat(
            [self.conv_sim(x), self.conv_sim(x.flip([1, 2])).flip([2])], 1
        )
        y_inr = torch.cat(
            [self.conv_inr(x), self.conv_inr(x.flip([1, 2])).flip([2])], 1
        )

        yact = self.activation(y)
        y_sim_act = self.activation(y_sim)  # * y_sim
        y_inr_act = self.activation(y_inr)

        y_pred = self.softplus(
            self.deconv(yact) + self.deconv_inr(y_inr_act) + self.deconv_sim(y_sim_act)
        )
        return y_pred

    def predict(self, seq_bp):

        if self.use_cuda:
            self.cuda()

        seq = sequences.sequence_to_encoding(
            seq_bp,
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
        seq_bp = seq_bp[325:-325]

        if self.use_cuda:
            pred0 = self(torch.FloatTensor(seq)[None, :, :].transpose(1, 2).cuda())
            pred0 = pred0.detach().cpu().numpy()
        else:
            pred0 = self(torch.FloatTensor(seq)[None, :, :].transpose(1, 2))
            pred0 = pred0.detach().numpy()

        index = ["Coordinate", "Sequence"]
        lines = {}
        lines["Coordinate"] = list(range(len(seq_bp)))
        lines["Sequence"] = list(seq_bp)

        for key in self.targeti_map.keys():
            lines["Prediciton " + key] = pred0[0, self.targeti_map[key], 325:-325]

        for key in self.targeti_rev_map.keys():
            lines["Prediciton rev strand " + key] = pred0[
                0, self.targeti_rev_map[key], 325:-325
            ]

        df = pd.DataFrame.from_dict(lines, orient="index")
        return df

    def basepair_contribution_transcription_initiation(
        self, seq, targeti, motifnames_original
    ):

        halfwindow = int((len(seq) - 650) / 2)

        tss_contr = {}
        seqs = []
        if self.use_cuda:
            seqt = torch.FloatTensor(seq)[None, :, :].transpose(1, 2).cuda()
        else:
            seqt = torch.FloatTensor(seq)[None, :, :].transpose(1, 2)
        preact_motif = torch.cat(
            [self.conv(seqt), self.conv(seqt.flip([1, 2])).flip([2])], 1
        )
        preact_inr = torch.cat(
            [self.conv_inr(seqt), self.conv_inr(seqt.flip([1, 2])).flip([2])], 1
        )
        preact_sim = torch.cat(
            [self.conv_sim(seqt), self.conv_sim(seqt.flip([1, 2])).flip([2])], 1
        )
        postact_motif = self.activation(preact_motif)
        postact_inr = self.activation(preact_inr)
        postact_sim = self.activation(preact_sim)

        seqcontri_exp = []
        seqcontri_bymotifs_exp = []
        seqcontri_inr_exp = []
        seqcontri_sim_exp = []
        for targeti in range(10):
            self.zero_grad()
            postact_motif_detached = postact_motif.detach()
            postact_motif_detached.requires_grad = True
            postact_inr_detached = postact_inr.detach()
            postact_inr_detached.requires_grad = True
            postact_sim_detached = postact_sim.detach()
            postact_sim_detached.requires_grad = True

            pred = self.activation(
                self.deconv(postact_motif_detached)
                + self.deconv_inr(postact_inr_detached)
                + self.deconv_sim(postact_sim_detached)
            )

            center = int(pred.shape[2] / 2)
            predexp = (
                10
                ** (
                    pred[:, targeti, center - halfwindow : center + halfwindow]
                    / np.log(10)
                )
                - 1
            )
            (predexp).sum().backward(retain_graph=True)

            self.zero_grad()
            seqt = seqt.detach()
            seqt.requires_grad = True
            preact_motif = torch.cat(
                [self.conv(seqt), self.conv(seqt.flip([1, 2])).flip([2])], 1
            )
            postact_motif = self.activation(preact_motif)
            ((postact_motif) ** 2 * postact_motif_detached.grad).sum().backward(
                retain_graph=True
            )
            seqcontri_exp.append(
                (seqt.grad * seqt.data).sum(axis=1).cpu().detach().numpy()
                - (seqt.grad * (1 - seqt.data)).sum(axis=1).cpu().detach().numpy() / 3
            )

            motifact_seq = []
            for i in range(20):
                self.zero_grad()
                seqt = seqt.detach()
                seqt.requires_grad = True
                preact_motif = torch.cat(
                    [self.conv(seqt), self.conv(seqt.flip([1, 2])).flip([2])], 1
                )

                postact_motif = self.activation(preact_motif)
                (
                    postact_motif[:, i, :] ** 2 * postact_motif_detached.grad[:, i, :]
                ).sum().backward()
                motifact_seq.append(
                    (seqt.grad * seqt.data).sum(axis=1).cpu().detach().numpy()
                    - (seqt.grad * (1 - seqt.data)).sum(axis=1).cpu().detach().numpy()
                    / 3
                )
            seqcontri_bymotifs_exp.append(motifact_seq)

            tss_contr[targeti] = {}
            for motif_n, motif in enumerate(motifnames_original):

                tss_contr[targeti][motif] = seqcontri_bymotifs_exp[targeti][motif_n][0][
                    325:-325
                ]

        return tss_contr

    def basepair_contribution_motif(self, seq, targeti, motifnames_original):

        seqts = []
        motif_contr = {}
        if self.use_cuda:
            seqt = torch.FloatTensor(seq)[None, :, :].transpose(1, 2).cuda()
        else:
            seqt = torch.FloatTensor(seq)[None, :, :].transpose(1, 2)

        for j in range(20):
            seqts.append(seqt)

        seqt = torch.cat(seqts, axis=0)
        seqt.requires_grad = True
        preact_motif = torch.cat(
            [self.conv(seqt), self.conv(seqt.flip([1, 2])).flip([2])], 1
        )

        postact_motif = self.activation(preact_motif)
        if self.use_cuda:
            (
                (postact_motif[:, :, :] * torch.eye(20).cuda()[:, :, None]) ** 2
            ).sum().backward()
        else:
            ((postact_motif[:, :, :] * torch.eye(20)[:, :, None]) ** 2).sum().backward()

        motifact_seq = (
            (seqt.grad * seqt.data).sum(axis=1).cpu().detach().numpy()
            - (seqt.grad * (1 - seqt.data)).sum(axis=1).cpu().detach().numpy() / 3
        )[..., 325:-325]

        motif_contr[targeti] = {}

        for motif_n, motif in enumerate(motifnames_original):
            motif_contr[targeti][motif] = motifact_seq[motif_n, :]

        return motif_contr

    def interpret(self, seq_bp, targeti="FANTOM_CAGE", reverse_strand=False):

        if reverse_strand:
            targeti = self.targeti_rev_map[targeti]
        else:
            targeti = self.targeti_map[targeti]

        if self.use_cuda:
            self.cuda()

        seq = sequences.sequence_to_encoding(
            seq_bp,
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
        seq_bp = seq_bp[325:-325]

        motifnames_original = [
            "SP-",
            "ETS-",
            "CREB-",
            "NFY-",
            "YY1-",
            "U1 snRNP+",
            "Long Inr+",
            "NRF1+",
            "ZNF143-",
            "TATA-",
            "SP+",
            "ETS+",
            "CREB+",
            "NFY+",
            "YY1+",
            "U1 snRNP-",
            "Long Inr-",
            "NRF1-",
            "ZNF143+",
            "TATA+",
        ]

        if self.use_cuda:
            pred0 = self(torch.FloatTensor(seq)[None, :, :].transpose(1, 2).cuda())
        else:
            pred0 = self(torch.FloatTensor(seq)[None, :, :].transpose(1, 2))

        tss_contr = self.basepair_contribution_transcription_initiation(
            seq, targeti, motifnames_original
        )
        motif_contr = self.basepair_contribution_motif(
            seq, targeti, motifnames_original
        )

        if self.use_cuda:
            preact_motif = torch.cat(
                [
                    self.conv(
                        torch.FloatTensor(seq)[None, :, :].transpose(1, 2).cuda()
                    ),
                    self.conv(
                        torch.FloatTensor(seq)[None, :, :]
                        .transpose(1, 2)
                        .flip([1, 2])
                        .cuda()
                    ).flip([2]),
                ],
                1,
            )

            postact_motif = self.activation(preact_motif)
            postact_motif = postact_motif

            preact_inr = torch.cat(
                [
                    self.conv_inr(
                        torch.FloatTensor(seq)[None, :, :].transpose(1, 2).cuda()
                    ),
                    self.conv_inr(
                        torch.FloatTensor(seq)[None, :, :]
                        .transpose(1, 2)
                        .flip([1, 2])
                        .cuda()
                    ).flip([2]),
                ],
                1,
            )

            postact_inr = self.activation(preact_inr)
            postact_inr = postact_inr
            effects_inr = self.deconv_inr(postact_inr).detach().cpu().numpy()

            preact_sim = torch.cat(
                [
                    self.conv_sim(
                        torch.FloatTensor(seq)[None, :, :].transpose(1, 2).cuda()
                    ),
                    self.conv_sim(
                        torch.FloatTensor(seq)[None, :, :]
                        .transpose(1, 2)
                        .flip([1, 2])
                        .cuda()
                    ).flip([2]),
                ],
                1,
            )

            postact_sim = self.activation(preact_sim)
            postact_sim = postact_sim
            effects_sim = self.deconv_sim(postact_sim).detach().cpu().numpy()

            w = self.deconv.weight[0, :, :].cpu().detach().numpy().max(axis=1)
            mw = self.conv.weight.cpu().detach().numpy().max(axis=2).max(axis=1)

        else:
            preact_motif = torch.cat(
                [
                    self.conv(torch.FloatTensor(seq)[None, :, :].transpose(1, 2)),
                    self.conv(
                        torch.FloatTensor(seq)[None, :, :].transpose(1, 2).flip([1, 2])
                    ).flip([2]),
                ],
                1,
            )

            postact_motif = self.activation(preact_motif)
            postact_motif = postact_motif

            preact_inr = torch.cat(
                [
                    self.conv_inr(torch.FloatTensor(seq)[None, :, :].transpose(1, 2)),
                    self.conv_inr(
                        torch.FloatTensor(seq)[None, :, :].transpose(1, 2).flip([1, 2])
                    ).flip([2]),
                ],
                1,
            )

            postact_inr = self.activation(preact_inr)
            postact_inr = postact_inr
            effects_inr = self.deconv_inr(postact_inr).detach().cpu().numpy()

            preact_sim = torch.cat(
                [
                    self.conv_sim(torch.FloatTensor(seq)[None, :, :].transpose(1, 2)),
                    self.conv_sim(
                        torch.FloatTensor(seq)[None, :, :].transpose(1, 2).flip([1, 2])
                    ).flip([2]),
                ],
                1,
            )

            postact_sim = self.activation(preact_sim)
            postact_sim = postact_sim
            effects_sim = self.deconv_sim(postact_sim).detach().numpy()

            w = self.deconv.weight[0, :, :].detach().numpy().max(axis=1)
            mw = self.conv.weight.detach().numpy().max(axis=2).max(axis=1)

        w = w * np.concatenate([mw, mw])
        inds = np.argsort(-w)

        motifact_offsets = np.zeros(len(motifnames_original))

        colors = [self.colordict[m] for m in motifnames_original]
        motif_activation = {}
        for i in inds:
            if not motifnames_original[i] in ["Long Inr+", "Long Inr-"]:
                motifactivation = postact_motif[0, i, 325:-325].detach().cpu().numpy().T
                motif_activation[motifnames_original[i]] = motifactivation - np.min(
                    motifactivation
                )

        dweight = self.deconv.weight.cpu().detach().numpy()
        effects_motifs = {}
        for i in inds:
            effects_motif = np.convolve(
                postact_motif[0, i, :].detach().cpu().numpy(),
                dweight[targeti, i, ::-1],
                mode="same",
            )[325:-325]
            effects_motifs[motifnames_original[i]] = effects_motif
            if motifnames_original[i] == "Long Inr+":
                effects_longinr = effects_motif
            elif motifnames_original[i] == "Long Inr-":
                effects_longinr_rev = effects_motif

        effect_inr = effects_inr[0, targeti, 325:-325]
        effect_sim = effects_sim[0, targeti, 325:-325]

        effect_inr = effect_inr + effects_longinr
        effect_inr = effect_inr - np.mean(effect_inr)
        effect_sim = effect_sim - np.mean(effect_sim)

        effect_motif = (
            np.vstack(list(effects_motifs.values())).sum(axis=0)
            - effects_longinr
            - effects_longinr_rev
        )
        effect_final = (
            effect_inr
            + effect_sim
            + effect_motif
            + (effects_longinr_rev - np.mean(effects_longinr_rev))
        )

        pred0 = pred0.detach().cpu().numpy()[0, :, 325:-325]

        lines = {}
        lines["Coordinate"] = list(range(len(seq_bp)))
        lines["Sequence"] = list(seq_bp)
        for i in range(20):
            if not motifnames_original[i] in ["Long Inr+", "Long Inr-"]:
                lines[motifnames_original[i] + " motif effect"] = effects_motifs[
                    motifnames_original[i]
                ]
                lines[motifnames_original[i] + " motif activation"] = motif_activation[
                    motifnames_original[i]
                ]

        lines["Sum of motif effect"] = effect_motif
        lines["Sum of initiator effect"] = effect_inr
        lines["Sum of trinucleotide effect"] = effect_sim
        lines["Sum of total effect"] = effect_final
        lines["Sum of motif effect after activation"] = pred0[targeti, :]

        for motif in tss_contr[targeti]:
            if not motif in ["Long Inr+", "Long Inr-"]:
                lines[motif + " TSS contribution"] = tss_contr[targeti][motif]

        for motif in motif_contr[targeti]:
            if not motif in ["Long Inr+", "Long Inr-"]:
                lines[motif + " motif contribution"] = motif_contr[targeti][motif]

        df = pd.DataFrame.from_dict(lines, orient="index")
        return df


if __name__ == "__main__":
    from docopt import docopt
    import sys

    doc = """
    Puffin outputs transcription inititation signal prediciton for the input genome sequence.
    
    Usage:
    puffin_predict coord [options] <coordinate>
    puffin_predict sequence [options] <fasta_file_path>
    puffin_predict region [options] <tsv_file>
    
    Options:
    -h --help        Show this screen.
    --no_interpretation     leave only Puffin prediction
    --target <target> experimental method targets (FANTOM_CAGE, ENCODE_CAGE, ENCODE_RAMPAGE, GRO_CAP, PRO_CAP) 
    --both_strands       generate predicion for the opposite strand
    --cuda     use cuda to generate the prediction
    """

    if len(sys.argv) == 1:
        sys.argv.append("-h")

    arguments = docopt(doc)

    if arguments["--target"]:
        targeti = arguments["--target"]
    else:
        targeti = "FANTOM_CAGE"
    print("making " + targeti + " prediction")

    if arguments["--cuda"]:
        use_cuda = True
    else:
        use_cuda = False

    ###load model###
    net = Puffin(use_cuda)

    ###load genome###
    genome_path = "./resources/hg38.fa"
    genome = selene_sdk.sequences.Genome(input_path=genome_path)

    seq_list = []
    name_list = []
    if arguments["coord"]:
        chrm, poss = arguments["<coordinate>"].split(":")
        strand = arguments["<coordinate>"][-1]
        start, end = poss[:-1].split("-")
        start = int(start)
        end = int(end)

        if strand == "-":
            offset = 1
            strand_ = "minus"
        else:
            offset = 0
            strand_ = "plus"

        seq_bp = genome.get_sequence_from_coords(
            chrm, start + offset, end + offset, strand
        )

        if len(seq_bp) < 651:
            print(
                "Minimum input sequence lenght should be > 651 bps, current sequence length is "
                + str(len(seq_bp))
                + " bps"
            )
            quit()
        else:
            seq_list.append(seq_bp)
            name = "puffin_" + chrm + "_" + str(start) + "_" + str(end) + "_" + strand_
            name_list.append(name)

    if arguments["sequence"]:
        fasta_file = open(arguments["<fasta_file_path>"], "r")
        for line in fasta_file:
            if line[0] == ">":
                name = line[1:-1]

            else:
                seq_bp = line
                if len(seq_bp) < 651:
                    print(
                        "Minimum input sequence lenght should be > 651 bps, current sequence length is "
                        + str(len(seq_bp))
                        + " bps"
                    )
                    continue
                else:
                    seq_list.append(seq_bp)

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

            seq_bp = genome.get_sequence_from_coords(
                chrm, start + offset, end + offset, strand
            )

            if len(seq_bp) < 651:
                print(
                    "Minimum input sequence lenght should be > 651 bps, current sequence length is "
                    + str(len(seq_bp))
                    + " bps"
                )
                quit()
            else:
                seq_list.append(seq_bp)
                name = (
                    "puffin_" + chrm + "_" + str(start) + "_" + str(end) + "_" + strand_
                )
                name_list.append(name)

    for seq_bp, name in zip(seq_list, name_list):
        if arguments["--no_interpretation"]:
            df = net.predict(seq_bp)
        else:
            df = net.interpret(seq_bp, targeti=targeti)
        df.to_csv(name + ".csv")
        print(name + " Done!")

        if arguments["--both_strands"]:

            df = net.interpret(seq_bp, targeti=targeti, reverse_strand=True)
            df.to_csv(name + ".csv")
            print("Reverse strand done!")
