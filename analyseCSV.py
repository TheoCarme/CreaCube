import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_parser():
    parser = argparse.ArgumentParser("analyse CSV")
    parser.add_argument(
        "-i",
        "--input_csv",
        type=str,
        default=r"F:\Results\p659\p659.csv",
        help="Path to your input csv file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=r"F:\Results",
        help="Path to your output directory.",
    )
    
    return parser


if __name__ == '__main__':

    args = make_parser().parse_args()
    
    df = pd.read_csv(args.input_csv, header=7, encoding = 'ansi', low_memory=False)
    
    columns = [["xc", "yc", "Score de confiance"], ["xc.1", "yc.1", "Score de confiance.1"],\
    ["xc.2", "yc.2", "Score de confiance.2"], ["xc.3", "yc.3", "Score de confiance.3"]]
    
    for i in range(4) :
    
        df[columns[i][0]] = df[columns[i][0]].diff()
        df[columns[i][1]] = df[columns[i][1]].diff()
        df[columns[i][2]] = df[columns[i][2]].pct_change()    
        df[columns[i][2]] = df[columns[i][2]].multiply(100)
    
    
    ax1 = df.plot.line(x=0, y=columns[0][0], color='orange')
    ax1 = df.plot.line(x=0, y=columns[0][1], color='purple', ax=ax1)
    plt.title("Spatial variations of the white cube over time")
    ax2 = df.plot.line(x=0, y=columns[0][2], color='blue')
    plt.title("Confidence variations of the white cube over time")
    
    ax3 = df.plot.line(x=0, y=columns[1][0], color='orange')
    ax3 = df.plot.line(x=0, y=columns[1][1], color='purple', ax=ax3)
    plt.title("Spatial variations of the red cube over time")
    ax4 = df.plot.line(x=0, y=columns[1][2], color='blue')
    plt.title("Confidence variations of the red cube over time")
    
    ax5 = df.plot.line(x=0, y=columns[2][0], color='orange')
    ax5 = df.plot.line(x=0, y=columns[2][1], color='purple', ax=ax5)
    plt.title("Spatial variations of the first dark cube over time")
    ax6 = df.plot.line(x=0, y=columns[2][2], color='blue')
    plt.title("Confidence variations of the first dark cube over time")
    
    ax7 = df.plot.line(x=0, y=columns[3][0], color='orange')
    ax7 = df.plot.line(x=0, y=columns[3][1], color='purple', ax=ax7)
    plt.title("Spatial variations of the second dark cube over time")
    ax8 = df.plot.line(x=0, y=columns[3][2], color='blue')
    plt.title("Confidence variations of the second dark cube over time")
    
    # df.plot.hist(column=["Score de confiance"])
    
    
    plt.show()
    
    # print(list(df.index.values))