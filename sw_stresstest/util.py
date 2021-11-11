import subprocess
import time

import matplotlib.pyplot as plt


def get_process_time():
    return "%.2f mins" % (time.process_time() / 60)


def savefig(sname, bar=False):
    barstr = "_bar" if bar else ""
    git_rev_hash = (
        subprocess.check_output("git rev-parse HEAD".split()).decode("utf-8").strip()
    )
    plotname = f"{sname}{barstr}_{git_rev_hash}.png"
    print(plotname)
    plt.savefig("plots/" + plotname)
    # plt.savefig('plots/' + plotname.replace('png', 'eps'))


# end


def setup_matplotlib():
    # plt.style.use('fivethirtyeight')
    # plt.style.use('ggplot')
    from cycler import cycler

    _cmap = plt.get_cmap("tab20")
    _cycler = cycler(
        color=[_cmap(i / 10) for i in range(10)] + [_cmap(3 / 12), _cmap(5 / 12)]
    ) + cycler(marker=[4, 5, 6, 7, "d", "o", ".", 4, 5, 6, 7, "d"])
    plt.rc("font", **{"size": 11})  # , 'sans-serif': ['Computer Modern Sans Serif']})
    plt.rc("axes", prop_cycle=_cycler, titlesize="xx-large", grid=True, axisbelow=True)
    plt.rc("grid", linestyle=":")
    plt.rc("figure", titlesize="xx-large")
    plt.rc("savefig", dpi=200)
    return _cycler


def draw_figlegend(fig, right=0.85, legend_title=None):
    # Remove duplicate label by using only 1 axes
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc=7, title=legend_title)
    fig.subplots_adjust(right=right)
