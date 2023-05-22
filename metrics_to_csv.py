#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from glob import glob
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="main path for tensorboard files", default=os.getcwd())
    parser.add_argument("--convert_step_to_epoch", type=bool, help="Convert steps to epoch", default=True)
    parser.add_argument("--output", type=str, help="Output Folder", default="")

    args = parser.parse_args()
    accs = {}
    losses= {}

    inpath = os.path.abspath(args.input)
    events = glob('**/events.out.tfevents.*', root_dir=inpath, recursive=True)
    accums = [(event, EventAccumulator(os.path.join(inpath, event)).Reload()) for event in events ]
    for (event, accum) in accums:
        if ("yolov5" in event):
            continue
        scalar_tags: list[str] = accum.Tags()['scalars']
        scalar_tags.remove('epoch')

        df = pd.DataFrame()
        emap = {}
        for epoch in accum.Scalars('epoch'):
            e = int(epoch.value)
            emap[epoch.step] = e

        for st in scalar_tags:
            for item in accum.Scalars(st):
                iep = emap[item.step]
                v = item.value
                try:
                    d = df.loc[iep, st]
                    df.loc[iep, st] = np.mean(d, v)
                except:
                    df.loc[iep, st] = v

        df.index.name = 'epoch'
        title = event.replace("\\", "/").rsplit("/", 1)[0].replace("/", " - ")
        out_dest = os.path.dirname(os.path.join(inpath, event))
        df.to_csv(path_or_buf=os.path.join(out_dest, "output.csv"))
        df.iloc[0:21].loc[:, ["val_loss", "val_acc", "train_loss_epoch", "train_acc_epoch"]].plot()
        plt.title(title)
        plt.grid(True)
        plt.ticklabel_format(axis='x', useOffset=False, style='plain')
        plt.xlim(0, 19)

        out_img = os.path.join(out_dest, "output.png")
        plt.savefig(out_img)
        if args.output != "":
            out_dir = os.path.abspath(args.output)
            df.to_csv(path_or_buf=os.path.join(out_dir, f"{title}.csv"))
            plt.savefig(os.path.join(out_dir, f"{title}.png"))

        accs[title] = df.loc[:, "val_acc"].max()
        losses[title] = df.loc[:, "val_loss"].min()

    print(accs)
    print(losses)
if __name__ == '__main__':
    main()

