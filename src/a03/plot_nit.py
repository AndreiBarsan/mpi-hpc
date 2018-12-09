
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

def main():

    data = """
n,m,w,nit_nat,nit_4c,err_nodes,err_38
8, 8 & 0.8 & 37 & 40 & 1.32344 & 1.30484	\\
16, 16 & 0.8 & 32 & 35 & 0.0723393 & 0.104943	\\
32, 32 & 0.8 & 28 & 32 & 0.0043547 & 0.0112814	\\
64, 64 & 0.8 & 25 & 31 & 0.000269573 & 0.00134882	\\
128, 128 & 0.8 & 22 & 30 & 1.68046e-05 & 0.000144203	\\
256, 256 & 0.8 & 18 & 30 & 1.04632e-06 & 1.99648e-05	\\
8, 8 & 0.9 & 29 & 33 & 1.32344 & 1.30484	\\
16, 16 & 0.9 & 26 & 29 & 0.0723393 & 0.104943	\\
32, 32 & 0.9 & 23 & 27 & 0.00435471 & 0.0112814	\\
64, 64 & 0.9 & 21 & 26 & 0.000269574 & 0.00134882	\\
128, 128 & 0.9 & 18 & 26 & 1.68066e-05 & 0.000144201	\\
256, 256 & 0.9 & 16 & 25 & 1.04736e-06 & 1.99622e-05	\\
8, 8 & 1 & 22 & 27 & 1.32344 & 1.30484	\\
16, 16 & 1 & 20 & 24 & 0.0723393 & 0.104943	\\
32, 32 & 1 & 18 & 23 & 0.00435471 & 0.0112814	\\
64, 64 & 1 & 17 & 23 & 0.000269575 & 0.00134882	\\
128, 128 & 1 & 16 & 23 & 1.68079e-05 & 0.0001442	\\
256, 256 & 1 & 16 & 23 & 1.04986e-06 & 1.99583e-05	\\
8, 8 & 1.1 & 18 & 24 & 1.32344 & 1.30484	\\
16, 16 & 1.1 & 19 & 31 & 0.0723393 & 0.104943	\\
32, 32 & 1.1 & 20 & 32 & 0.00435471 & 0.0112814	\\
64, 64 & 1.1 & 20 & 32 & 0.000269575 & 0.00134882	\\
128, 128 & 1.1 & 20 & 32 & 1.68079e-05 & 0.0001442	\\
256, 256 & 1.1 & 20 & 32 & 1.04988e-06 & 1.99583e-05	\\
8, 8 & 1.2 & 19 & 32 & 1.32344 & 1.30484	\\
16, 16 & 1.2 & 24 & 43 & 0.0723393 & 0.104943	\\
32, 32 & 1.2 & 25 & 45 & 0.00435471 & 0.0112814	\\
64, 64 & 1.2 & 25 & 43 & 0.000269575 & 0.00134882	\\
128, 128 & 1.2 & 26 & 43 & 1.68079e-05 & 0.0001442	\\
256, 256 & 1.2 & 26 & 43 & 1.04989e-06 & 1.99583e-05	\\
    """
    string_io = StringIO(data.replace('&', ','))
    df = pd.read_csv(string_io, sep=",")

    df = df[df['n'] == 64]
    print(df['nit_nat'])
    ax = df.plot('w', 'nit_nat', label="$n_{it}$ (natural order)")
    ax = df.plot('w', 'nit_4c', ax=ax, label="$n_{it}$ (four-color order)")
    ax.set(
        xlabel='$\omega$', ylabel='$n_{it}$'
    )
    plt.tight_layout()
    out = '../../results/plots/a03-sor-256'
    # plt.savefig(out + '.png')
    # plt.savefig(out + '.eps')
    plt.show()

    print(df)


if __name__ == '__main__':
    main()
