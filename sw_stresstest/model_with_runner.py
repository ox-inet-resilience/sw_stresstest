import random
import os
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from .model import Model
from .util import get_process_time

# logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler(filename='logg.txt')])


def info(sname):
    print(sname, get_process_time())


def transpose(x):
    return zip(*x)


def run_parallel(_func, arr, args):
    procs = []
    for x in arr:
        proc = mp.Process(target=_func, args=(x,) + args)
        proc.start()
        procs.append(proc)
    for p in procs:
        p.join()


# begin useful helper measures
def get_extent_of_contagion(out, nbanks, truncate=True, contagion=False):
    # See Gai-Kapadia 2010
    # remainder is |\mathcal{B} \setminus \mathcal{D}|
    assert nbanks > 0
    num_defaults = sum(out[1:])
    if truncate:
        ret = num_defaults / nbanks
        if ret > 1:
            print("eoc is greater than 1:", ret)
        # Truncation for the foundation paper
        # paper to not exceed 100%.
        return min(1, ret)
    if contagion:
        # Only contagion defaults
        return num_defaults - out[1]
    # Number of bank failures
    return num_defaults


def plot_custom_errorbar_plot(
    x, y, std, use_marker=True, color=None, marker=None, label=""
):
    if color is None:
        ax = plt.gca()
        _cc = next(ax._get_lines.prop_cycler)
        color, marker = _cc.values()
    if use_marker:
        (l,) = plt.plot(
            x, y, marker=marker, markerfacecolor="none", color=color, label=label
        )
    else:
        (l,) = plt.plot(x, y, marker=",", color=color, label=label)
    y = np.array(y)
    std = np.array(std)
    plt.fill_between(x, y - std, y + std, color=color, alpha=0.4)
    return l


def mul_by_100(x):
    return [i * 100 for i in x]


def div_by_1m(x):
    return [i / 1000_000 for i in x]


def plot_aeocs(
    _ax,
    X,
    aeocs,
    stdeocs,
    eba_eose,
    label="",
    use_marker=True,
    draw_eba_eose_marker=False,
    plotting_mode="BANKDEFAULTS",
    truncate_y_axis=True,
    contagion=False,
    nbanks=None,
    ylabel=None,
):
    plt.sca(_ax)
    ax = plt.gca()
    # plt.gca().set_xlim(left=0)
    _cc = next(ax._get_lines.prop_cycler)
    color, marker = _cc.values()
    if plotting_mode == "ASSETLOSS":
        plt.ylim(-0.02, 6.5)
        aeocs = div_by_1m(aeocs)
        stdeocs = div_by_1m(stdeocs)
        if eba_eose:
            eba_eose = div_by_1m(eba_eose)
    elif plotting_mode == "RFF":
        pass
        # plt.ylim(bottom=-0.02)
    else:
        if truncate_y_axis:
            plt.ylim(-2, 105)
            aeocs = mul_by_100(aeocs)
            stdeocs = mul_by_100(stdeocs)
            eba_eose = mul_by_100(eba_eose)
    if True:  # force to plot error bar
        _l = plot_custom_errorbar_plot(
            X, aeocs, stdeocs, use_marker, color, marker, label
        )
    else:
        # without error bar
        (_l,) = plt.plot(
            X, aeocs, marker=marker, markerfacecolor="none", color=color, label=label
        )
    # plot aeo-se caused by initial defaults
    _l_eba = None
    if eba_eose:
        (_l_eba,) = plt.plot(
            X,
            eba_eose,
            color="darkgrey",
            marker="o",
            markerfacecolor="none",
            markersize=10,
        )
        if draw_eba_eose_marker:
            plt.plot(
                X,
                eba_eose,
                marker=marker,
                markerfacecolor="none",
                color=color,
                linestyle="none",
            )
    if plotting_mode == "ASSETLOSS":
        if contagion:
            plt.ylabel("Contagious asset loss (trn euros)")
        else:
            plt.ylabel("Asset loss (trn euros)")
    elif plotting_mode == "RFF":
        plt.ylabel("Resolution financing fund contribution $F$ (bln euros)")
    else:
        if truncate_y_axis:
            plt.ylabel("Bank defaults $\\mathbb{E}$ (%)")
        else:
            if contagion:
                plt.ylabel("Contagious bank failures")
            else:
                plt.ylabel("Bank failures")
    return _l, _l_eba


def plot_stacked_bar(x, data, width=0.8, color=None, bar_fn=None):
    if bar_fn is None:
        bar_fn = plt.bar
    # plot the first one
    label, y = data[0]
    if color:  # for the first one only
        _bar = bar_fn(x, y, width, color=color, label=label)
    else:
        _bar = bar_fn(x, y, width, label=label)
    bars = [_bar]
    new_bottom = np.array(y)
    # plot the rest
    for label, _y in data[1:]:
        _bar = bar_fn(x, _y, width, bottom=new_bottom, label=label)
        bars.append(_bar)
        new_bottom += _y
    return bars


# Model with additional methods to help run multiple simulation sets.
class ModelWithRunner(Model):
    def __init__(self):
        random.seed(1337)
        np.random.seed(1337)
        super().__init__()
        os.makedirs("plots", exist_ok=True)

    def set_params_postdefault_contagions_only(self):
        # post-default contagion only
        # contagions:
        # 1. post-default firesale
        self.parameters.POSTDEFAULT_FIRESALE_CONTAGION = True
        self.parameters.PREDEFAULT_FIRESALE_CONTAGION = False
        # 2. post-default funding pull
        # 4. possible post-default repo
        self.parameters.POSTDEFAULT_PULLFUNDING_CONTAGION = True
        # 3. exposure loss
        self.parameters.INTERBANK_LOSS_GIVEN_DEFAULT = 0
        self.parameters.HAIRCUT_CONTAGION = False

        self.parameters.TRADABLE_ASSETS_ON = True
        self.parameters.N_GOV_BONDS = 21
        self.parameters.N_CORP_BONDS = 24
        self.parameters.N_EQUITIES = 24
        self.parameters.N_OTHERTRADABLES = 24

        self.parameters.BANK_LEVERAGE_MIN = 0.03

        # 1.25% price drop with 5% asset sold
        self.parameters.PRICE_IMPACTS = self.parameters.price_impact_fn_pi(0.0125)

    def _run_one_sim(
        self,
        truncate,
        param=None,
        apply_param=None,
        use_rwa=True,
        contagion=False,
    ):
        if apply_param is not None:
            apply_param(param)

        out = self.run_schedule()
        if self.parameters.PLOT_ASSET_LOSS:
            # asset loss
            if contagion:
                return (
                    self.asset_loss - self.initial_asset_loss,
                    self.initial_asset_loss,
                )
            return self.asset_loss, self.initial_asset_loss
        elif hasattr(self.parameters, "PLOT_RFF") and self.parameters.PLOT_RFF:
            # Used only in bail-in simulations BF6c
            # resolution financing fund
            return sum(b.cumulative_rff for b in self.banks), None

        # bank defaults extent of contagion
        # eoc is f^d(n) in the paper
        eoc = get_extent_of_contagion(out, len(self.banks), truncate, contagion)
        if truncate:
            _eba_eose = out[1] / self.nbanks
            if _eba_eose < 0.05:
                _eba_eose = 0
        else:
            _eba_eose = out[1]
        return eoc, _eba_eose

    def _run_one_sim_set(
        self,
        truncate,
        iteration,
        apply_param,
        params,
        check_previous_defaults=True,
        use_rwa=True,
        contagion=False,
    ):
        random.seed(iteration)

        self.reset_networks()

        eoc_set, EBA_eose = transpose(
            [
                self._run_one_sim(truncate, param, apply_param, use_rwa, contagion)
                for param in params
            ]
        )
        return eoc_set, EBA_eose

    def common_procedure_11(
        self,
        ax,
        params,
        apply_param,
        X,
        use_rwa,
        second_param=None,
        N=20,
        sname=None,
        truncate=True,  # whether to truncate eoc to 1
        contagion=False,
        print_defaults=True,
        check_previous_defaults=True,
        color=None,
        # plotting-specific params
        additional_label="",
        draw_eba_eose=True,
        use_marker=True,
        draw_eba_eose_marker=False,
        do_legend=False,
    ):
        """
        params
        apply_param
        X
        use_rwa
        second_param
        N: number of repetition
        additional_label: additional label on the plot
        do_legend: whether to draw legend
        print_defaults: whether number of defaults is put on the plot
        color: line color
        sname: simulation name
        parralel: use multiprocessing
        """
        EBA_eose = 0

        if self.parallel:
            import multiprocessing as mp
            import sys

            def except_hook(exctype, value, tb):
                sys.__excepthook__(exctype, value, tb)
                for p in mp.active_children():
                    p.terminate()

            sys.excepthook = except_hook
        aeocs = []
        stdeocs = []

        # calibration, precompute initial defaults due to sw shock
        # this will precompute self.defaulted_banks_from_sw_shock
        # TODO this is ugly hack
        apply_param(params[-1])
        self.find_defaulted_banks_from_sw_shock()

        if self.parallel:
            manager = mp.Manager()

            def _fn(_i):
                eoc_set, EBA_eose = self._run_one_sim_set(
                    truncate,
                    _i,
                    apply_param,
                    params,
                    check_previous_defaults,
                    use_rwa,
                    contagion,
                )
                par_eoc_sets.append(eoc_set)
                par_EBA_eose.append(EBA_eose)

            if N < 200:
                par_eoc_sets = manager.list()
                par_EBA_eose = manager.list()
                run_parallel(_fn, range(N), ())
                eocs_across_param = transpose([i for i in par_eoc_sets])
                EBA_eose = par_EBA_eose[0]
            else:
                eoc_sets = []
                epochs = np.array_split(np.arange(N), 5)
                for epoch in epochs:
                    par_eoc_sets = manager.list()
                    par_EBA_eose = manager.list()
                    run_parallel(_fn, epoch, ())
                    eoc_sets += [i for i in par_eoc_sets]
                eocs_across_param = transpose(eoc_sets)
                EBA_eose = par_EBA_eose[0]
        else:
            _sim_sets = [
                self._run_one_sim_set(
                    truncate,
                    i,
                    apply_param,
                    params,
                    check_previous_defaults,
                    use_rwa,
                    contagion,
                )
                for i in range(N)
            ]
            eocs_across_param = transpose([i[0] for i in _sim_sets])
            EBA_eose = _sim_sets[0][1]

        if self.parameters.PLOT_ASSET_LOSS:
            # Plotting asset losses
            ave_als = []
            std_als = []
            EBA_al = EBA_eose  # variable rename for clarity
            for asset_losses in eocs_across_param:
                length = len(asset_losses)
                ave_al = sum(asset_losses) / length
                std_al = np.std(asset_losses) / np.sqrt(length)
                ave_als.append(ave_al)
                std_als.append(std_al)
            if not draw_eba_eose:
                EBA_al = None
            ave_als_line, eba_als_line = plot_aeocs(
                ax,
                X,
                ave_als,
                std_als,
                EBA_al,
                additional_label,
                plotting_mode="ASSETLOSS",
                truncate_y_axis=truncate,
                contagion=contagion,
                use_marker=use_marker,
            )
            # aeocs, stdeocs, probability, initial eose
            self.outcome = [ave_als, std_als, EBA_al]
            return ave_als_line, eba_als_line
        elif hasattr(self.parameters, "PLOT_RFF") and self.parameters.PLOT_RFF:
            # Plotting resolution financing fund
            ave_rffs = []
            std_rffs = []
            for rffs in eocs_across_param:
                length = len(rffs)
                ave_rff = sum(rffs) / length
                std_rff = np.std(rffs) / np.sqrt(length)
                ave_rffs.append(ave_rff)
                std_rffs.append(std_rff)
            ave_rffs_line, _ = plot_aeocs(
                ax,
                X,
                ave_rffs,
                std_rffs,
                None,
                additional_label,
                plotting_mode="RFF",
                truncate_y_axis=truncate,
                contagion=contagion,
                use_marker=use_marker,
            )
            # aeocs, stdeocs, probability, initial eose
            self.outcome = [ave_rffs, std_rffs, None]
            return ave_rffs_line, None

        # Plotting bank defaults
        for eocs in eocs_across_param:
            _eocs = [e for e in eocs if e >= 0.05]

            # this is |\mathcal{C}| in the paper
            nose = len(_eocs)  # number of systemic event
            if nose == 0:
                aeoc = 0
                stdeoc = 0
            else:
                aeoc = sum(_eocs) / nose
                stdeoc = np.std(_eocs) / np.sqrt(nose)
            aeocs.append(aeoc)
            stdeocs.append(stdeoc)

        if not draw_eba_eose:
            EBA_eose = None
        aeocs_line, eba_line = plot_aeocs(
            ax,
            X,
            aeocs,
            stdeocs,
            EBA_eose,
            additional_label,
            use_marker=use_marker,
            draw_eba_eose_marker=draw_eba_eose_marker,
            truncate_y_axis=truncate,
            contagion=contagion,
        )

        # if sname is not None:
        #    with open('plots/_' + sname + '.json', 'w') as outfile:
        #        json.dump(_out, outfile)

        if do_legend:
            plt.legend()
        # aeocs, stdeocs, probability, initial eose
        self.outcome = [aeocs, stdeocs, EBA_eose]
        return aeocs_line, eba_line

    def set_EBA_2018_stress_test_scenario(self, sw_scale=None):
        # system-wide shock
        if sw_scale is None:
            sw_scale = self.parameters.SYSTEMWIDE_SHOCK_SCALE
        if sw_scale <= 0:
            # There is no system-wide shock happening for this case
            return
        for i in range(len(self.banks)):
            b = self.banks[i]
            # self.change_initial_external(b, -self.EBA_shocks_external[i])
            shock = min(0, self.EBA_shocks_RWCR[i]) * sw_scale
            shock_lev = min(0, self.EBA_shocks_lev[i]) * sw_scale
            lev = b.get_leverage()
            lev_adverse = lev + shock_lev

            self.shift_RWCR(b, shock, use_external=True)

            # shift leverage after external shock
            lev_after_1st_shock = b.get_leverage()
            lev_den = b.leverage_constraint.get_leverage_denominator()
            b.DeltaA += lev_den * (1 - lev_after_1st_shock / lev_adverse)

    def find_defaulted_banks_from_sw_shock(self):
        self.defaulted_banks_from_sw_shock = [
            i for i in range(len(self.banks)) if self.banks[i].is_insolvent()
        ]

    def calculate_system_vulnerability(self):
        vulns = []
        vuln = 0
        total_initial_equity = sum(b.get_ledger().initial_equity for b in self.banks)
        for b in self.banks:
            E_i = b.get_ledger().initial_equity
            _h = min(1, (E_i - b.get_equity_valuation()) / E_i)
            _w = E_i / total_initial_equity
            vuln += _w * _h
            vulns.append(_h)
        return vuln, vulns
