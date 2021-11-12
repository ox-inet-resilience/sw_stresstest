import random
import time
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from sw_stresstest.model import (
    GSIB_surcharges,
    systemic_risk_buffer,
    CCyB_buffer,
    RHO_CCB,
)
from sw_stresstest.util import (
    savefig as old_savefig,
    setup_matplotlib,
)
from sw_stresstest.model_with_runner import (
    ModelWithRunner,
    info,
    run_parallel,
    mul_by_100,
)

# OSIB_surcharges_2017 taken from https://www.eba.europa.eu/risk-analysis-and-data/other-systemically-important-institutions-o-siis-/2017
from sw_stresstest.data.EU_OSIB_surcharges_2017 import OSIB_surcharges_2017
from sw_stresstest.parameters import GSIBs

_cycler = setup_matplotlib()

# To be used at all simulation FF
# Note: if the length of price_impacts is changed, the variable `indices` in
# run_FF11 is no longer accurate. It is hardcoded with the assumption that the
# 0th element is 0% price impact, and the 5th element is the 5% price impact.
NPOINTS = 11
price_impacts = np.linspace(0.0, 0.1, NPOINTS)
shocks = np.linspace(0, 2, NPOINTS)
usability = np.linspace(0, 1, NPOINTS)
buffer_mul = np.linspace(0, 8, NPOINTS)
PI = 0.05
NSIM = 10


def get_stde(x):
    return np.std(x) / np.sqrt(len(x))


def hypothenuse(a, b):
    return np.sqrt(a ** 2 + b ** 2)


def savefig(x, bar=False):
    old_savefig(f"{x}_N{NSIM}", bar)


def get_rho_combined_buffer(bname):
    _country = bname[:2]

    rho_GSIB = GSIB_surcharges.get(bname, 0) / 100
    rho_CCyB = CCyB_buffer.get(_country, 0) / 100
    rho_SRB = systemic_risk_buffer.get(_country, 0) / 100
    rho_OSIB = OSIB_surcharges_2017[bname] / 100
    rho_CB = RHO_CCB + max(rho_GSIB, rho_OSIB, rho_SRB) + rho_CCyB
    return rho_CB, rho_GSIB, rho_CCyB


class FoundationSimulation(ModelWithRunner):
    def __init__(self):
        self.TIC = time.time()
        random.seed(1337)
        np.random.seed(1337)
        super().__init__()

    def set_constraints(self, rwa=True, lev=True, lcr=True):
        # IMPORTANT: this is to ensure the network stays fresh for each new session
        self.reset_networks()

        # set constraints
        self.parameters.BANK_RWA_ON = rwa
        self.parameters.BANK_LEVERAGE_ON = lev
        self.parameters.BANK_LCR_ON = lcr

        self._count = 0  # for counting the progress of the simulation
        self._total_count = 0

    def print_count(self):
        self._count += 1
        print(
            self._count,
            "/",
            self._total_count,
            "%.2f mins" % ((time.time() - self.TIC) / 60),
        )

    def isa_GSIB(self, b):
        if b is None:
            return False
        if hasattr(b, "isa_GSIB"):
            return b.isa_GSIB
        if not self.parameters.BAILIN_GSIB_ONLY:
            return True
        gsibs = self.custom_SIBs if self.custom_SIBs is not None else GSIBs
        return b.get_name() in gsibs

    # foundation paper
    def common_initialisation_FF(
        self,
        pi,
        RWCR_weight=0.5,
        leverage_weight=0.5,
        isFF6=False,
        FF6_label=None,
        FF6_ratio=None,
        FF6_multiplier=None,
        FF6_RWCR_y=1,
        FF6_leverage_y=1,
        set_LGD=True,
        LCR_weight=0.5,
    ):
        self.parameters.TRADABLE_ASSETS_ON = True
        self.parameters.PRICE_IMPACTS = self.parameters.price_impact_fn_pi(pi)
        self.parameters.POSTDEFAULT_PULLFUNDING_CONTAGION = True
        self.parameters.POSTDEFAULT_FIRESALE_CONTAGION = True
        # the parameter FF3_* is used only at simulation FF3
        if set_LGD:
            self.parameters.INTERBANK_LOSS_GIVEN_DEFAULT = 1.0  # set LGD to 100%
        self.parameters.ENDOGENOUS_LGD_ON = False
        self.parameters.PREDEFAULT_FIRESALE_CONTAGION = True
        self.parameters.PREDEFAULT_PULLFUNDING_CONTAGION = True

        # RWA
        self.parameters.RWCR_FLTF = rho_M = self.parameters.RHO_M_STANDARD
        if isFF6 and FF6_ratio == "RWA" and "req" in FF6_label:
            self.parameters.RWCR_FLTF = rho_M = FF6_multiplier * rho_M
        # Lev
        LEV_M_STANDARD = 0.03
        self.parameters.BANK_LEVERAGE_MIN = lev_min = LEV_M_STANDARD
        if isFF6 and FF6_ratio == "leverage" and "req" in FF6_label:
            self.parameters.BANK_LEVERAGE_MIN = lev_min = FF6_multiplier * lev_min
        # LCR
        LCR_S = 1  # 100%
        self.parameters.BANK_LCR_BUFFER = LCR_weight * LCR_S
        self.parameters.BANK_LCR_TARGET = (
            self.parameters.BANK_LCR_BUFFER + self.parameters.BANK_LCR_EXCESS_TARGET
        )

        self.initialise()
        for b in self.banks:
            _name = b.get_name()

            rho_combined_buffer, rho_GSIB, _ = get_rho_combined_buffer(_name)

            # leverage
            # 0.012778 is the average of GSIBs' rho_GSIB
            leverage_BF = (rho_GSIB if (_name in GSIBs) else 0.012778) / 2
            if isFF6 and FF6_ratio == "leverage" and "buf" in FF6_label:
                FF6_leverage_y = FF6_multiplier
            b.leverage_buffer = lev_min + leverage_weight * FF6_leverage_y * leverage_BF
            b.leverage_target = (
                b.leverage_buffer + self.parameters.BANK_LEVERAGE_EXCESS_TARGET
            )

            # RWCR
            if isFF6 and FF6_ratio == "RWA" and "buf" in FF6_label:
                FF6_RWCR_y = FF6_multiplier
            b.RWCR_buffer = rho_M + RWCR_weight * FF6_RWCR_y * rho_combined_buffer
            b.RWCR_target = b.RWCR_buffer + self.parameters.BANK_RWA_EXCESS_TARGET

            assert b.leverage_target >= b.leverage_buffer, (
                b.leverage_target,
                b.leverage_buffer,
                leverage_weight,
            )
            assert b.RWCR_target >= b.RWCR_buffer, (
                b.RWCR_target,
                b.RWCR_buffer,
                RWCR_weight,
            )

            if isFF6:
                # FF6 specific
                if "req" in FF6_label:
                    if FF6_ratio == "RWA":
                        offset = rho_M - self.parameters.RHO_M_STANDARD
                        self.shift_RWCR(b, offset)
                    elif FF6_ratio == "leverage":
                        offset = lev_min - LEV_M_STANDARD
                        self.shift_leverage(b, offset)
                elif "buf" in FF6_label:
                    if FF6_ratio == "RWA":
                        offset = FF6_RWCR_y * rho_combined_buffer - rho_combined_buffer
                        self.shift_RWCR(b, offset)
                    elif FF6_ratio == "leverage":
                        offset = FF6_leverage_y * leverage_BF - leverage_BF
                        self.shift_leverage(b, offset)
            else:
                # Used in FF3, FF5, FF7, FF8
                if FF6_RWCR_y > 1:
                    offset = FF6_RWCR_y * rho_combined_buffer - rho_combined_buffer
                    self.shift_RWCR(b, offset)

                # Used in FF4
                if FF6_leverage_y > 1:
                    offset = FF6_leverage_y * leverage_BF - leverage_BF
                    self.shift_leverage(b, offset)
        # For measurement purpose, just like self.bank_defaults_this_round
        if self.parameters.PLOT_ASSET_LOSS:
            self.total_asset_initial = self.get_system_total_asset()

    def _do_bar_plot(self, labels, fn, N, pi=PI, figsize=(8, 5), legend_title=None):
        # common for FF11, FF8
        print("bar")

        def _bar(mode, label):
            resolution = mode == 1
            do_multiply = isinstance(mode, str) and "x" in mode.split()[0]
            constraint_type = mode.split()[1] if do_multiply else None
            multiplication = int(mode.split()[0][0]) if do_multiply else 1
            eocs = []
            EBA_eose = 0
            for i in range(N):
                random.seed(i)
                self.reset_networks()
                fn(label, resolution, do_multiply, constraint_type, multiplication)(pi)
                eoc, eose_i = self._run_one_sim(True)
                if eoc >= 0.05:
                    eocs.append(eoc)
                    EBA_eose = eose_i
            if len(eocs) == 0:
                return 0, 0, 0
            return np.mean(eocs), get_stde(eocs), EBA_eose

        _parout = mp.Manager().dict()

        def __fn(m, l):
            _parout[(m, l)] = _bar(m, l)
            print(m, l)

        modes = (0, "2x lev", "2x rw")
        fig = plt.figure(figsize=figsize)
        x = np.arange(len(modes))
        x_offset = 0
        plt.ylim(0, 105)
        for label in labels:
            run_parallel(__fn, modes, (label,))
            _out = [_parout[(m, label)] for m in modes]
            aeocs, stds, eba_eoses = list(zip(*_out))
            # Use percentage instead
            aeocs = mul_by_100(aeocs)
            stds = mul_by_100(stds)
            _w = 0.2
            plt.bar(x + x_offset, aeocs, _w, yerr=stds, capsize=3, label=label)
            # plt.bar(x + x_offset, eba_eoses, _w - 0.01, color='lightgrey')
            x_offset += _w
            print()
        plt.ylabel("Bank defaults $\\mathbb{E}$ (%)")
        disorderly = "Disorderly\nliquidation\n"
        plt.xticks(
            x + _w,
            (
                f"{disorderly}(Basel III)",
                f"{disorderly}(2x lev\n buffer)",
                f"{disorderly}(2x rw\n buffer)",
            ),
        )
        plt.tight_layout()
        fig.subplots_adjust(right=0.77)
        fig.legend(loc=7, title=legend_title)

    def run_FF10(self, name_ext=""):
        sname = "FF10" + name_ext
        info(sname)

        self.parameters.MARGIN_CALL_ON = True
        self.set_constraints()

        def fn(label, resolution, do_multiply, constraint_type, multiplication):
            self.parameters.HEDGEFUNDS_ON = "HF" in label
            self.parameters.HAIRCUT_CONTAGION = "HF" in label
            self.parameters.ASSET_MANAGER_ON = (" F " in label) or ("(F)" in label)

            def _fn(pi):
                if do_multiply:
                    if constraint_type == "lev":
                        self.common_initialisation_FF(PI, FF6_leverage_y=multiplication)
                    elif constraint_type == "rw":
                        self.common_initialisation_FF(PI, FF6_RWCR_y=multiplication)
                    else:
                        raise Exception("constraint not supported")
                else:
                    self.common_initialisation_FF(PI)
                self.common_initial_stress_test_scenario(resolution, rwcr=True)

            return _fn

        labels = (
            "Banks (B)",
            "B + Hedge\nFunds (HF)",
            "B +\nNon-Leveraged\nFunds (F)",
            "B + F + HF",
        )

        # TODO ugly hack
        # this is to calculate the defaulted banks from sw shock
        self.common_initialisation_FF(PI)
        self.common_initial_stress_test_scenario(rwcr=True)
        self.find_defaulted_banks_from_sw_shock()

        # bar plot
        self._do_bar_plot(
            labels,
            fn,
            NSIM,
            figsize=(8, 5),
            legend_title="Types of\ninstitutions\nturned on:",
        )
        savefig(sname, True)

    def run_FF11(self, name_ext="", resolution=False, rwa_y=1, lev_y=1):
        sname = "FF11" + name_ext
        info(sname)

        self.set_constraints()

        self.parameters.ASSET_MANAGER_ON = True
        self.parameters.HEDGEFUNDS_ON = True
        self.parameters.HAIRCUT_CONTAGION = True

        individuals = [
            "Overlap.\nportfolio\ncontagion (O)",
            "Exposure\nloss\ncontagion (E)",
            "Funding\ncontagion (F)",
            "Collateral\ncontagion (C)",
        ]
        doubles = ["O&E", "O&F", "O&C", "F&E"]
        tri_fours = ["O&E&F&C"]
        modes = individuals + doubles + tri_fours

        # FF11C
        modes_c = doubles + tri_fours

        self._total_count = len(modes)

        fig = plt.figure()
        all_outcome = {}  # used in FF11c

        def _set_EFC(label):
            self.parameters.INTERBANK_LOSS_GIVEN_DEFAULT = 1 if "E" in label else 0  # E
            self.parameters.FUNDING_CONTAGION_INTERBANK = "F" in label  # F
            self.parameters.MARGIN_CALL_ON = "C" in label  # C

        def _f(label, vary_shock=False):
            do_price_impact = "O" in label

            def fn(x):
                _set_EFC(label)
                sw_mul = x if vary_shock else 1
                pi = PI if vary_shock else x
                _pi = pi if do_price_impact else 0
                # set_LGD is False because LGD is already set by _set_EFC. We
                # don't want to override that.
                self.common_initialisation_FF(
                    _pi, set_LGD=False, FF6_RWCR_y=rwa_y, FF6_leverage_y=lev_y
                )
                self.common_initial_stress_test_scenario(
                    resolution, rwcr=True, sw_scale=sw_mul
                )

            ax = plt.gca()
            _x = shocks if vary_shock else price_impacts
            _xticks = shocks if vary_shock else price_impacts * 100
            aeocs_line, _ = self.common_procedure_11(
                ax,
                _x,
                fn,
                _xticks,
                self.parameters.BANK_RWA_ON,
                N=NSIM,
                additional_label=label,
                sname=sname,
                use_marker=True,
            )
            _label = label if "&" in label else label[label.find("(") + 1]
            all_outcome[_label] = list(self.outcome)

            ax.set_xlabel("Price impact (%)")
            if vary_shock:
                ax.set_xlabel("Initial shock relative to EBA 2018 scenario")
            self.print_count()
            return aeocs_line

        print(modes)
        vary_shock = 0
        for mode in modes:
            _f(mode, vary_shock)

        fig.legend(loc=7, title="Contagion\nmechanisms:")
        fig.subplots_adjust(right=0.73)
        plt.title(
            "Contagion-free resolution" if resolution else "Disorderly liquidation"
        )
        savefig(sname)

        sname += "c"
        for modec in ["minus", "over"]:
            plt.clf()
            info(sname + modec)

            # For explanation of this `indices` variable, see the comment just
            # before the definition of price_impacts.
            indices = (0, 5)

            def get_diff_std(label):
                diff_aeocs = []
                diff_stds = []
                for idx in indices:
                    # aeocs (0) - EBA eose (1)
                    sum_aeocs = sum(
                        all_outcome[m][0][idx] - all_outcome[m][-1][idx]
                        for m in label.split("&")
                    )
                    join_contagion_aeocs = (
                        all_outcome[label][0][idx] - all_outcome[label][-1][idx]
                    )
                    sum_stds = np.sqrt(
                        sum(all_outcome[m][1][idx] ** 2 for m in label.split("&"))
                    )
                    if modec == "minus":
                        diff_aeocs.append(join_contagion_aeocs - sum_aeocs)
                        diff_stds.append(
                            hypothenuse(all_outcome[label][1][idx], sum_stds)
                        )
                    else:
                        # calculate ratio instead
                        f = (
                            join_contagion_aeocs / sum_aeocs
                            if sum_aeocs > 0
                            else (5 if join_contagion_aeocs > 0 else 0)
                        )
                        diff_aeocs.append(f)
                        join_err = (
                            all_outcome[label][1][idx] / join_contagion_aeocs
                            if join_contagion_aeocs > 0
                            else 0
                        )
                        sum_err = sum_stds / sum_aeocs if sum_aeocs > 0 else 0
                        diff_stds.append(f * hypothenuse(join_err, sum_err))

                if modec == "over":
                    # truncate to be at least 1
                    return [max(da, 1) for da in diff_aeocs], diff_stds
                else:
                    # truncate diff to be within 0 and 1
                    return [max(min(da, 1), 0) for da in diff_aeocs], diff_stds

            x = np.arange(len(indices), dtype=float)
            _w = 0.12
            ax = plt.gca()
            ax.set_prop_cycle(_cycler[4:])
            offset = 0

            def _sort(x):
                _a = list(np.argsort(-x, kind="stable"))
                return [_a.index(i) for i in range(len(modes_c))]

            diffs = np.array([get_diff_std(l)[0] for l in modes_c])
            orders = np.array([_sort(x) for x in diffs.T]).T
            for i, label in enumerate(modes_c):
                diff, diff_std = get_diff_std(label)
                offset = _w * np.array(orders[i])
                plt.bar(x + offset, diff, _w, yerr=diff_std, capsize=3, label=label)
                offset += _w
            if modec == "over":
                plt.ylabel("Amplification (joint over parts)")
            else:
                plt.ylabel("Excess fraction of defaults (joint minus parts)")
            plt.ylim(bottom=-0.04)
            plt.xticks(x + _w * 2, ("Price impact 0%", "Price impact 5%"))
            fig.legend(loc=7, title="Contagion\nmechanisms:")
            fig.subplots_adjust(right=0.77)
            plt.title(
                "Contagion-free resolution" if resolution else "Disorderly liquidation"
            )
            savefig(f"{sname}_{modec}")
        # reset params
        self.parameters.ASSET_MANAGER_ON = False
        self.parameters.HEDGEFUNDS_ON = False
        self.parameters.HAIRCUT_CONTAGION = False
        # reset EFC
        self.parameters.INTERBANK_LOSS_GIVEN_DEFAULT = 1  # E
        self.parameters.FUNDING_CONTAGION_INTERBANK = True  # F
        self.parameters.MARGIN_CALL_ON = False  # C

    def common_initial_stress_test_scenario(
        self, resolution=False, rwcr=True, sw_scale=1
    ):
        if resolution:
            self.parameters.INTERBANK_LOSS_GIVEN_DEFAULT = 0
            self.parameters.POSTDEFAULT_FIRESALE_CONTAGION = False
            self.parameters.POSTDEFAULT_PULLFUNDING_CONTAGION = False
        self.set_EBA_2018_stress_test_scenario(sw_scale=sw_scale)
        if self.parameters.PLOT_ASSET_LOSS:
            # this is for measurement purpose
            self.initial_asset_loss = (
                self.total_asset_initial - self.get_system_total_asset()
            )

    def common_procedure_FF3(
        self,
        sname,
        fn,
        N,
        labels=None,
        figsize=(10, 5),
        do_resolution=True,
        fig_axs=None,
        do_title=True,
        legend_loc=7,
        do_usability=False,
        use_marker=True,
        legend_title=None,
        do_vary_shock=False,
        draw_eba_eose_marker=False,
        do_buffer_mul=False,
        ratio_FF14=None,
        adjust_right=0.87,
        do_eba_eose_label=False,
    ):
        if fig_axs is None:
            if do_resolution:
                fig, axs = plt.subplots(1, 2, figsize=figsize)
            else:
                fig = plt.figure(figsize=figsize)
        else:
            fig, axs = fig_axs

        if labels is None:
            labels = ("0%", "25%", "50%", "75%", "100%")
        self._total_count = len(labels) * (2 if do_resolution else 1)

        def _f(label, resolution=False):
            if do_resolution:
                ax = axs[1 if resolution else 0]
            else:
                ax = plt.gca()
            if do_usability:
                _x = usability
                xticks = 100 * _x
            elif do_vary_shock:
                _x = shocks
                xticks = _x
            elif do_buffer_mul:
                _x = buffer_mul
                xticks = _x
            else:
                _x = price_impacts
                xticks = 100 * _x
            aeocs_line, eba_line = self.common_procedure_11(
                ax,
                _x,
                fn(label, resolution),
                xticks,
                self.parameters.BANK_RWA_ON,
                N=NSIM,
                additional_label=label,
                sname=sname,
                use_marker=use_marker,
                draw_eba_eose_marker=draw_eba_eose_marker,
            )
            # Only used in FF14_bar
            if "FF14" in sname and "b" not in sname:
                _r = (ratio_FF14 == "RWA") if ratio_FF14 else self.parameters.BANK_RWA_ON
                self.outcome_FF14[resolution][label.split()[0], _r] = self.outcome
            ax.set_xlabel("Price impact (%)")
            if do_usability:
                ax.set_xlabel("Usability of buffers (%)")
            elif do_vary_shock:
                ax.set_xlabel("Initial shock relative to EBA 2018 scenario")
            elif do_buffer_mul:
                ax.set_xlabel("Scale of regulatory buffer size")
            self.print_count()
            return aeocs_line, eba_line

        def _do_one_sim(resolution):
            lines = [_f(i, resolution) for i in labels]
            print()
            __out = [l[0] for l in lines]
            if lines[0][1] is not None:  # EBA eose line
                __out.append(lines[0][1])
            return __out

        lines = _do_one_sim(False)
        if do_resolution:
            lines = _do_one_sim(True)
            if do_title:
                axs[0].set_title("Disorderly liquidation")
                axs[1].set_title("Contagion-free resolution")
            fig.subplots_adjust(right=adjust_right)
        else:
            fig.subplots_adjust(right=0.8)
        if do_eba_eose_label:
            labels += ("Microprudential\nstress test\noutcome",)
        plt.figlegend(lines, labels, loc=legend_loc, title=legend_title)
        savefig(sname)

    def _modified_common_procedure_FF3(
        self, sname, fn, N, fig_axs, legend_title=None, legend_loc=7, labels=None
    ):
        # Used in FF12a, FF12b, FF17
        if labels is None:
            labels = ("0%", "25%", "50%", "75%", "100%")
        self._total_count = len(labels) * 2
        fig, _axs = fig_axs

        def _f(label, vary_shock=False):
            ax = _axs[1 if vary_shock else 0]
            if vary_shock:
                _x = shocks
                xticks = _x
            else:
                _x = price_impacts
                xticks = 100 * _x
            aeocs_line, _ = self.common_procedure_11(
                ax,
                _x,
                fn(label, vary_shock),
                xticks,
                self.parameters.BANK_RWA_ON,
                N=NSIM,
                additional_label=label,
                sname=sname,
            )
            ax.set_xlabel("Price impact (%)")
            if vary_shock:
                ax.set_xlabel("Initial shock relative to EBA 2018 scenario")
            self.print_count()
            return aeocs_line

        def _do_one_sim(vary_shock):
            lines = [_f(i, vary_shock) for i in labels]
            print()
            return lines

        lines = _do_one_sim(False)
        lines = _do_one_sim(True)
        fig.subplots_adjust(right=0.87)
        plt.figlegend(lines, labels, loc=legend_loc, title=legend_title)
        savefig(sname)

    def make_contour_plot(self, xs, ys, fn, xlabel=None, ylabel=None):
        plt.figure()
        X, Y = np.meshgrid(xs, ys)
        zs = np.array([fn(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        plt.contourf(X, Y, Z)
        plt.colorbar()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.tight_layout()

    def run_FF12a(self, name_ext="", rwa_y=1, lev_y=1, resolution=True):
        sname = "FF12a" + name_ext
        info(sname)

        fig, axs = plt.subplots(2, 2, figsize=(12.5, 10))
        plt.suptitle(
            "Contagion-free resolution" if resolution else "Disorderly liquidation"
        )
        fig.subplots_adjust(top=0.93)
        # previously 0.88, 0.88
        plt.gcf().text(
            0.88, 0.58, "RW capital buffer (leverage ratio + LCR off)", rotation=270
        )
        # previously 0.88, 0.43
        plt.gcf().text(
            0.88, 0.13, "Leverage buffer (RW capital ratio + LCR off)", rotation=270
        )
        # RW buffer
        self.set_constraints(1, 0, 0)
        ratio = "RWA"

        def fn(label, vary_shock):
            print(label, vary_shock, ratio)

            def _fn(x):
                weight = 1 - int(label[:-1]) / 100
                pi = PI if vary_shock else x
                sw_mul = x if vary_shock else 1
                if ratio == "RWA":
                    self.common_initialisation_FF(
                        pi, RWCR_weight=weight, FF6_RWCR_y=rwa_y
                    )
                else:
                    self.common_initialisation_FF(
                        pi, leverage_weight=weight, FF6_leverage_y=lev_y
                    )
                self.common_initial_stress_test_scenario(
                    resolution, rwcr=ratio == "RWA", sw_scale=sw_mul
                )

            return _fn

        self._modified_common_procedure_FF3(
            sname, fn, NSIM, fig_axs=(fig, axs[0]), legend_title="Usability of\nbuffer:"
        )

        # leverage buffer
        self.set_constraints(0, 1, 0)
        ratio = "leverage"

        self._modified_common_procedure_FF3(
            sname, fn, NSIM, fig_axs=(fig, axs[1]), legend_title="Usability of\nbuffer:"
        )

    def run_FF12b(self, name_ext="", lev_y=1, resolution=True, all_constraints=True):
        sname = "FF12b" + name_ext
        info(sname)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(
            "Contagion-free resolution" if resolution else "Disorderly liquidation"
        )
        fig.subplots_adjust(top=0.93)
        if all_constraints:
            # previously 0.88, 0.7
            plt.gcf().text(0.875, 0.3, "RW + leverage + LCR buffer", rotation=270)
        else:
            # previously 0.88, 0.9
            plt.gcf().text(
                0.875,
                0.1,
                "LCR buffer (RW + leverage buffer usability fixed at 50%)",
                rotation=270,
            )

        self.set_constraints()

        def fn(label, vary_shock):
            def _fn(x):
                weight = 1 - int(label[:-1]) / 100
                pi = PI if vary_shock else x
                sw_mul = x if vary_shock else 1
                rw_weight = weight if all_constraints else 0.5
                lev_weight = weight if all_constraints else 0.5
                self.common_initialisation_FF(
                    pi,
                    RWCR_weight=rw_weight,
                    leverage_weight=lev_weight,
                    FF6_leverage_y=lev_y,
                    LCR_weight=weight,
                )
                self.common_initial_stress_test_scenario(resolution, sw_scale=sw_mul)

            return _fn

        if all_constraints:
            title = "Usability of\nbuffer:"
        else:
            title = "LCR\nusability:"
        self._modified_common_procedure_FF3(
            sname, fn, NSIM, fig_axs=(fig, axs), legend_title=title
        )

    def run_FF12c(self, name_ext="", resolution=True, rwa_y=1):
        sname = "FF12c" + name_ext
        info(sname)
        self.set_constraints()
        xs = usability * 100
        ys = price_impacts * 100

        def fn(x, y):
            weight = 1 - x / 100
            pi = y / 100
            self.common_initialisation_FF(
                pi,
                RWCR_weight=weight,
                leverage_weight=weight,
                FF6_RWCR_y=rwa_y,
                LCR_weight=weight,
            )
            self.common_initial_stress_test_scenario(resolution)
            self.find_defaulted_banks_from_sw_shock()
            eoc, _ = self._run_one_sim(True)
            return eoc

        self.make_contour_plot(
            xs, ys, fn, xlabel="Usability of buffers (%)", ylabel="Price impact (%)"
        )
        savefig(sname)

    def run_FF14(self, name_ext="", vary_shock=False, all_constraints=False):
        sname = "FF14" + name_ext
        info(sname)

        self.outcome_FF14 = {False: {}, True: {}}  # for bar
        fig, axs = plt.subplots(2, 2, figsize=(11, 10))
        plt.gcf().text(0.88, 0.74, "RW ratio", rotation=270)
        plt.gcf().text(0.88, 0.3, "Leverage ratio", rotation=270)

        # requirements only
        labels_req = "half req", "Basel III", "2x req"
        labels_buf = ("2x buffer", "4x buffer")
        if "3" in name_ext:
            labels_buf = ("2x buffer", "3x buffer", "4x buffer")
        labels = labels_req + labels_buf
        labels = ("Basel III",) + labels_buf + ("3x req", "4x req")
        labels = labels_req + ("3x req", "4x req")
        labels = ("Basel III",) + labels_buf

        # RW buffer
        if all_constraints:
            self.set_constraints()
        else:
            self.set_constraints(1, 0, 0)
        ratio = "RWA"

        multiplier_map = {"2x": 2, "half": 0.5, "3x": 3, "4x": 4, "Basel": 1}

        def fn(label, resolution):
            double_or_half = label.split(" ")[0]
            multiplier = multiplier_map[double_or_half]
            print(label, resolution, ratio)

            def _fn(x):
                pi = PI if vary_shock else x
                sw_mul = x if vary_shock else 1
                self.common_initialisation_FF(
                    pi,
                    isFF6=True,
                    FF6_label=label,
                    FF6_ratio=ratio,
                    FF6_multiplier=multiplier,
                )
                self.common_initial_stress_test_scenario(
                    resolution, ratio == "RWA", sw_scale=sw_mul
                )

            return _fn

        self.common_procedure_FF3(
            sname,
            fn,
            NSIM,
            labels=labels,
            fig_axs=(fig, axs[0]),
            do_vary_shock=vary_shock,
            draw_eba_eose_marker=1,
            ratio_FF14=ratio,
            legend_title="Regulatory\nbuffer size:",
        )

        # leverage buffer
        if all_constraints:
            self.set_constraints()
        else:
            self.set_constraints(0, 1, 0)
        ratio = "leverage"
        self.common_procedure_FF3(
            sname,
            fn,
            NSIM,
            labels=labels,
            fig_axs=(fig, axs[1]),
            do_vary_shock=vary_shock,
            draw_eba_eose_marker=1,
            ratio_FF14=ratio,
            do_title=False,
            legend_title="Regulatory\nbuffer size:",
        )

        plt.close()

    def run_FF15(
        self,
        name_ext="",
        ratio="all",
        rwa_y=1,
        lev_y=1,
        alter_lcr_only=False,
        vary_shock=False,
    ):
        sname = "FF15" + name_ext
        info(sname)
        if ratio == "RWA":
            self.set_constraints(1, 0, 0)
        elif ratio == "leverage":
            self.set_constraints(0, 1, 0)
        else:  # all
            self.set_constraints()

        def fn(label, resolution):
            target_offset = eval(label[:-1]) / 100

            def _fn(x):
                pi = PI if vary_shock else x
                sw_mul = x if vary_shock else 1
                self.common_initialisation_FF(
                    pi, FF6_RWCR_y=rwa_y, FF6_leverage_y=lev_y
                )
                if alter_lcr_only:
                    self.parameters.BANK_LCR_TARGET = (
                        self.parameters.BANK_LCR_BUFFER
                        + self.parameters.BANK_LCR_EXCESS_TARGET
                        + target_offset
                    )
                    self.common_initial_stress_test_scenario(
                        resolution, sw_scale=sw_mul
                    )
                    return
                for b in self.banks:
                    if ratio in ["RWA", "all"]:
                        b.RWCR_target = b.RWCR_buffer + target_offset
                    if ratio in ["leverage", "all"]:
                        b.leverage_target = b.leverage_buffer + target_offset
                self.common_initial_stress_test_scenario(resolution, sw_scale=sw_mul)

            return _fn

        if alter_lcr_only:
            _labels = ("5%", "10%", "25%", "50%")
        else:
            _labels = ("0.5%", "1%", "2%", "3%", "4%", "5%")
        self.common_procedure_FF3(
            sname,
            fn,
            NSIM,
            labels=_labels,
            figsize=(11.5, 5),
            do_vary_shock=vary_shock,
            legend_title="Excess of\ntarget above\nbuffer:",
            adjust_right=0.89,
        )

    def run_FF9(self, constraints=None, rwa_y=1, lev_y=1, name_ext=""):
        sname = "FF9" + name_ext
        info(sname)
        if constraints is None:
            self.set_constraints(1, 1, 1)
        else:
            self.set_constraints(*constraints)

        labels = ("System-wide\nstress test\noutcome",)

        def fn(label, resolution):
            def _fn(x):
                self.common_initialisation_FF(
                    PI, FF6_RWCR_y=rwa_y, FF6_leverage_y=lev_y
                )
                self.common_initial_stress_test_scenario(resolution, sw_scale=x)

            return _fn

        fig, axs = plt.subplots(1, 2, figsize=(11.5, 5))

        # To make sure the plot has orange color instead of blue
        next(axs[0]._get_lines.prop_cycler)
        next(axs[1]._get_lines.prop_cycler)

        self.common_procedure_FF3(
            sname,
            fn,
            NSIM,
            labels=labels,
            fig_axs=(fig, axs),
            adjust_right=0.83,
            do_vary_shock=True,
            do_eba_eose_label=True,
        )
        savefig(sname)

    def run_sim_groups(self, FF9_10=False, FF11=False, FF12_14_15=False):
        # FSWST
        print("Number of repetitions:", NSIM)
        if FF9_10:
            self.run_FF9(name_ext="_1")
            self.run_FF9(name_ext="_2", constraints=(1, 0, 0))
            self.run_FF9(name_ext="_3", constraints=(0, 1, 0))
            self.run_FF9(name_ext="_4", rwa_y=3)

            self.run_FF10(name_ext="_1")
            # The self.parameters value after FF10 is run is a bit messy, so better quit
            # the program right after running FF10.
            print("Total elapsed time:", time.process_time())
            exit()

        if FF11:
            # To be sure, FF11 should be run in its own simulation cluster because
            # its config is not reset after run. TODO: fix this.
            _n = 0
            lrs = [(1, 1), (2.5, 1)]
            for lev_y, rwa_y in lrs:
                _n += 1
                self.run_FF11(name_ext=f"_{_n}_({lev_y},{rwa_y})_liquidation_", rwa_y=rwa_y, lev_y=lev_y)
            print("Total elapsed time:", time.process_time())
            exit()

        if FF12_14_15:
            # self.run_FF12a(name_ext='_1')
            self.run_FF12a(name_ext="_2", resolution=False)
            self.run_FF12b(name_ext="_3")
            self.run_FF12b(name_ext="_4", all_constraints=False)

            self.run_FF14(name_ext="_3", all_constraints=True)
            self.run_FF14(name_ext="_4", all_constraints=True, vary_shock=True)

            # self.run_FF15(name_ext='_1')
            self.run_FF15(name_ext="_2", lev_y=3)
            # self.run_FF15(name_ext='_3', ratio='RWA')
            # self.run_FF15(name_ext='_4', vary_shock=True)
            # self.run_FF15(name_ext='_7', alter_lcr_only=True)

        print("Total elapsed time:", time.process_time())

if __name__ == "__main__":
    fs = FoundationSimulation()
    # This is a flag on whether to parallelize the NSIMS simulations via Python
    # multiprocessing.
    fs.parallel = True
    fs.run_sim_groups(FF9_10=True)
    # fs.run_sim_groups(FF11=True)
    # fs.run_sim_groups(FF12_14_15=True)
