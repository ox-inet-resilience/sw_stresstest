import random
import csv

import numpy as np

# import networkx as nx
from economicsl import Simulation

from resilience.contracts import (
    TradableAsset,
    AssetCollateral,
    Repo,
    Deposit,
    Other,
    Loan,
)
from resilience.markets import AssetMarket
from resilience.agents import Bank, Hedgefund, AssetManager

from .parameters import Parameters, DATA_YEAR, isequal_float
from .network import generate_montagna_l1_network, generate_montagna_l3_network

random.seed(1337)
np.random.seed(1337)

AssetType = Parameters.AssetType

RHO_CCB = 0.025  # https://www.esrb.europa.eu/national_policy/capital/html/index.en.html

GSIB_surcharges_2015 = {
    "UK48": 2.5,  # HSBC
    "UK49": 2.0,  # Barclays
    "DE15": 2.0,  # Deutsche Bank
    # 1.5 Credit Suisse
    "FR13": 1.0,  # Groupe BPCE
    "FR12": 1.0,  # Groupe Credit Agricole
    "NL31": 1.0,  # ING Groep
    "SE44": 1.0,  # Nordea
    "UK50": 1.0,  # Royal Bank of Scotland
    "ES38": 1.0,  # Banco Santander
    # Standard Chartered
    # UBS
    "IT28": 1.0,  # Unicredit Group
}

# from FSB 2017 list of global systemically important banks (G-SIBs)
GSIB_surcharges_2017 = {
    "UK46": 2.0,  # HSBC
    "UK45": 1.5,  # Barclays
    "DE17": 2.0,  # Deutsche Bank
    # 1.0 Credit Suisse
    # 'FR13'  # Groupe BPCE HAS BEEN REMOVED MAKE SURE TO DISABLE THIS!!!
    "FR12": 1.0,  # Groupe Credit Agricole
    "NL32": 1.0,  # ING Groep
    "SE41": 1.0,  # Nordea
    "UK48": 1.0,  # Royal Bank of Scotland
    "ES39": 1.0,  # Banco Santander
    # Standard Chartered
    # UBS
    "IT28": 1.0,  # Unicredit Group
}

GSIB_surcharges = GSIB_surcharges_2017 if DATA_YEAR == 2017 else GSIB_surcharges_2015

# CCyB data from https://www.esrb.europa.eu/national_policy/ccb/applicable/html/index.en.html
# As of 22 August 2018
CCyB_buffer = {
    "AT": 0,  # since 1 Jan 2016
    "BE": 0,  # since 1 Jan 2016
    "DE": 0,  # since 1 Jan 2016
    "DK": 0,  # since 1 Jan 2016
    "ES": 0,  # since 1 Jan 2016
    "FI": 0,  # since 16 Mar 2015
    "FR": 0,  # since 30 Dec 2015
    "HU": 0,  # since 1 Jan 2016
    "IE": 0,  # since 1 Jan 2016
    "IT": 0,  # since 1 Jan 2016
    "NL": 0,  # since 1 Jan 2016
    "NO": 0,  # 2.0 since 31 Dec 2017
    "PL": 0,  # since 1 Jan 2016
    "SE": 2.0,  # since 19 Mar 2017
    "UK": 0,  # 0.5 since 27 Jun 2018
}

# https://www.cesifo-group.de/ifoHome/facts/DICE/Banking-and-Financial-Markets/Banking/Bank-Regulation-and-Legal-Framework/appl-syst-risk-buffers/fileBinary/appl-syst-risk-buffers.pdf
# year: 2016
# in percentages
systemic_risk_buffer = {
    # Estonia 1.0
    # Slovak Republic 1.0
    "AT": 1.5,  # average of 1.0% - 2.0%
    "NL": 3,
}


def assert_equal(x, y):
    assert isequal_float(x, y), (x, y)


def read_csv(fname):
    # this function assumes the delimiter to be ' ' and 1 row to
    # be skipped
    with open(fname, "r") as _f:
        _reader = csv.reader(_f, delimiter=" ")
        next(_reader)  # skip header
        out = list(_reader)
    return out


NBANKS = 48 if DATA_YEAR == 2017 else 51


def generate_interbank_network_retry(asts, lias):
    try:
        return generate_montagna_l1_network(asts, lias)
    except AssertionError:
        # yeah sometimes this happens
        # rerun the network generation again
        return generate_montagna_l1_network(asts, lias)


class Model:
    def __init__(self):
        """Initialize the simulation"""
        self.simulation = None
        self.interbank_matrix = None
        self.govbonds_matrix = None
        self.corpbonds_matrix = None
        self.equities_matrix = None
        self.custom_SIBs = None
        self.nbanks = NBANKS
        # We don't exclude banks when using public data, because the data has
        # no nan's.
        self.banned_banks = []

    def get_time(self):
        return self.simulation.get_time()

    def run_schedule(self):
        """
        Runs the simulation with global parameters in Parameters.
        While agents act sequentially, the consequence of their actions is
        independent of their execution order.
        """
        self.alives = [self.check_for_alive()]
        num_defaults = [0]
        while self.get_time() < self.parameters.SIMULATION_TIMESTEPS:
            self.simulation.advance_time()

            # reset the number of banks that default this round
            self.simulation.bank_defaults_this_round = 0
            # shuffle the execution order just in case
            random.shuffle(self.allAgents)

            self.process_mailboxes()
            self.assetMarket.clear_the_market()
            for agent in self.allAgents:
                agent.act_fulfil_contractual_obligations()
            # This second process_mailboxes() is necessary so that all banks
            # receive cash from the contractual obligations (the substep just
            # above this) on time.
            self.process_mailboxes()
            for agent in self.allAgents:
                agent.act()
            for agent in self.allAgents:
                if agent.marked_as_default:
                    agent.trigger_default()

            # for analysis purpose
            self.alives.append(self.check_for_alive())
            num_defaults.append(self.simulation.bank_defaults_this_round)

        if self.parameters.PLOT_ASSET_LOSS:
            self.asset_loss = self.total_asset_initial - self.get_system_total_asset()
        return num_defaults

    def get_system_total_asset(self):
        # only banks for now
        return sum(b.get_ledger().get_asset_valuation() for b in self.banks)

    def process_mailboxes(self):
        self.simulation.process_postbox()
        # Check inboxes
        for agent in self.allAgents:
            agent.step()

    def setup_banks_balance_sheet(self):
        self.setup_banks_balance_sheet_with_public_data()

    def initialise(self, parameters=Parameters):
        """Initialises a particular scenario, with NBANKS banks, NBANKS Hedgefunds, and one AssetMarket
        """
        self.parameters = parameters
        self.simulation = Simulation()
        self.allAgents = []
        self.assetMarket = AssetMarket(self)

        self.banks = self.setup_banks()
        self.setup_banks_balance_sheet()
        self.setup_banks_interbank_network()

        # HF initialization
        hfs, Q = self.setup_hedgefunds()

        # Initialize asset managers
        ams, invs = self.setup_asset_managers()

        for i in range(len(self.banks)):
            if self.parameters.HEDGEFUNDS_ON:
                self.initRepo(self.banks[i], hfs[i], self.snl_rrepo[i] * Q)
                self.initRepo(self.banks[i], None, (1 - Q) * self.snl_rrepo[i])
            else:
                self.initRepo(self.banks[i], None, self.snl_rrepo[i])
            self.initRepo(None, self.banks[i], self.snl_repo[i])

        self.allAgents += self.banks + hfs + ams + invs

        self.hfs = hfs
        self.ams = ams

        for agent in self.allAgents:
            # set_initial_valuations() computes the initial equity for an agent
            agent.set_initial_valuations()

        self.calibrate_RWA_weight()

        # Calibrate LCR
        self.calibrate_LCR_weight_other()

        # Calculate initial leverage for HFs
        # It is necessary to do this after the initRepo step
        for hf in hfs:
            if hf.get_ledger().get_asset_valuation() == 0:
                hf.LEVERAGE_INITIAL = 0
                hf.uec_fraction_initial = 0
            else:
                hf.LEVERAGE_INITIAL = hf.get_leverage()
                hf.uec_fraction_initial = (
                    hf.get_ue_cash() / hf.get_ledger().get_asset_valuation()
                )
            # sanity check
            assert (
                hf.leverage_constraint.get_effective_min_leverage()
                <= hf.LEVERAGE_INITIAL
            )

        # Sanity checks
        if self.parameters.MARGIN_CALL_ON:
            # Skip sanity checks since margin call on initialisation will change the values anyway
            return
        if not self.parameters.DO_SANITY_CHECK:
            return
        self.do_sanity_check()

    def shift_RWCR(self, b, offset, RWA=None, use_external=False):
        # offset can be calculated by (new value) - (old value)
        if RWA is None:
            RWA = b.rwa_constraint.get_RWA()

        if use_external:
            # shift external asset instead
            rho = b.get_RWA_ratio()
            _weight = b.RWA_weights["external"]
            delta_ext = offset * RWA / (1 - (rho + offset) * _weight)
            is_enough = self.change_initial_external(b, delta_ext)
            if not is_enough:
                rho_final = rho + offset
                new_offset = rho_final - b.get_RWA_ratio()
                self.shift_RWCR(b, new_offset)
        else:
            alpha = -offset * RWA
            self.change_initial_liability(b, alpha)

    def shift_RWCR_deposit(self, b, offset):
        # This is used in bailin simulations.
        # This is just like shift_RWCR()'s
        # change_initial_liability, except that the pecking order
        # is to shift deposit first, then other liability.
        # The reason is that other liability is part of bailinable, and so
        # shouldn't be altered if we want to get consistent result.
        RWA = b.rwa_constraint.get_RWA()
        alpha = -offset * RWA
        ldg = b.get_ledger()
        # There are 2 deposit contracts, the first is not-bailinable, and
        # the second is bailinable. We only modify the first.
        nonbailinable_deposit = ldg.get_liabilities_of_type(Deposit)[0]
        new_deposit = nonbailinable_deposit.get_valuation("L") + alpha
        if new_deposit >= 0:
            nonbailinable_deposit.principal = new_deposit
            return
        raise Exception("Lowering liability by this much is not yet supported")
        # else 'deposit' is fully exhausted
        ldg.get_liabilities_of_type(Deposit)[0].principal = 0
        other_l = ldg.get_liability_valuation_of(Other)
        new_other = other_l + new_deposit
        if new_other >= 0:
            others = ldg.get_liabilities_of_type(Other)
            for o in others:
                o.set_amount(o.get_notional() * new_other / other_l)
        raise Exception("Lowering liability by this much is not yet supported")

    def shift_leverage(self, b, offset):
        _offset = offset
        alpha = -_offset * b.leverage_constraint.get_leverage_denominator()
        self.change_initial_liability(b, alpha)

    def change_initial_external(
        self, b, delta_ext, delta_ext_is_fraction_valuation=False
    ):
        _eas = [
            a
            for a in b.get_ledger().get_assets_of_type(TradableAsset)
            if a.get_asset_type() == AssetType.EXTERNAL1
        ]
        if len(_eas) == 0:
            return 0
        ext_asset = _eas[0]
        # It is always the notional that is being shocked
        # But for external asset, the valuation is always identical to notional
        ext_notional = ext_asset.get_valuation("A")
        if delta_ext_is_fraction_valuation:
            delta_ext *= ext_notional
        assert (
            delta_ext <= 0
        ), delta_ext  # i.e. only used for reducing assets that it must be negative
        shock_val = min(ext_notional, -delta_ext)
        ext_asset.quantity -= shock_val
        return ext_notional >= -delta_ext

    def change_initial_liability(self, b, delta_L):
        # helper function for the distance experiment and average RW capital ratio shift
        # with pecking order of Other, Deposit, Repo, Loan
        ldg = b.get_ledger()
        other_l = ldg.get_liability_valuation_of(Other)
        new_other = other_l + delta_L
        if new_other >= 0:
            os = ldg.get_liabilities_of_type(Other)
            if len(os) == 1:
                # There is only 1 other liability
                os[0].set_amount(new_other)
            else:
                # Usually when non-banks are turned on
                # NOTE it is sufficient to modify the notional without having
                # to convert the valuation into notional because the
                # revaluation_multiplier if new_other and other_l
                # cancels out each other
                for o in os:
                    o.set_amount(o.get_notional() * new_other / other_l)
            return
        # else 'other' is fully exhausted
        ldg.get_liabilities_of_type(Other)[0].set_amount(0)

        new_deposit = ldg.get_liability_valuation_of(Deposit) + new_other
        if new_deposit >= 0:
            # There is only 1 deposit
            ldg.get_liabilities_of_type(Deposit)[0].principal = new_deposit
            return
        # else 'deposit' is fully exhausted
        ldg.get_liabilities_of_type(Deposit)[0].principal = 0

        old_repo = ldg.get_liability_valuation_of(Repo)
        new_repo = old_repo + new_deposit
        if new_repo >= 0:
            # reduce proportionally
            repos = ldg.get_liabilities_of_type(Repo)
            assert len(repos) == 1, len(repos)
            for r in repos:
                r.principal = r.principal * new_repo / old_repo
            return
        raise Exception("Lowering liability by this much is not yet supported")

    def make_initial_defaults(self, n, check_previous_defaults=True):
        _out = []
        defaulted_banks = []
        if check_previous_defaults:
            defaulted_banks = self.defaulted_banks_from_sw_shock
        while len(_out) < n:
            x = random.randrange(0, self.nbanks)
            # make sure the banks are all different and not already defaulted earlier
            if x in defaulted_banks:
                continue
            if x not in _out:
                _out.append(x)
        return _out

    def make_some_banks_insolvent(
        self,
        ninitial_defaults=1,
        use_RWA=False,
        initial_defaults=None,
        check_previous_defaults=True,
    ):
        # make some banks insolvent/to bailin at the beginning
        # This code has existed since foundation simulations, but is only used
        # in bailin simulations
        # MAKE SURE TO NOT USE IT FOR FOUNDATION SIMULATIONS!
        if initial_defaults is None:
            initial_defaults = self.make_initial_defaults(
                ninitial_defaults, check_previous_defaults
            )
        BELOW_FLTF = 0.04  # how much lower beyond FLTF point
        for i in initial_defaults:
            b = self.banks[i]
            d = b.get_RWA_ratio_distance() if use_RWA else b.get_leverage_distance()
            offset = -(d + BELOW_FLTF) if d > 0 else 0
            if use_RWA:
                # This function is used instead of shift_RWCR because we do not
                # want to alter bailinable debt (other liability).
                self.shift_RWCR_deposit(b, offset)
            else:
                self.shift_leverage(b, offset)
        return initial_defaults

    def addExternalAsset(self, agent, assetType, quantity):
        """adds an external asset to an agent"""
        if agent is None:
            return
        if quantity > 0:
            agent.add(TradableAsset(agent, assetType, self.assetMarket, quantity))

    def initInterBankLoan(self, lender, borrower, principal):
        """
         lender:
            asset party

         borrower:
            liability party

         principal:
            principal
        """
        loan = Loan(lender, borrower, principal)
        lender.add(loan)
        borrower.add(loan)
        return loan

    def initRepo(self, lender, borrower, principal):
        """
        Similar to initInterBankLoan. If FUNDING_CONTAGION_INTERBANK is
        switched on, the repo is as expected. If it is switched off, two copies
        of the repo are made: each one points at one agent on one side, and at
        the 'null' agent on the other side. The 'null' agent pledges all
        necessary collateral immediately, never defaults, and pays immediately.

            lender:
                asset party (reverse-repo party)

            borrower:
                liability party (repo party)

            principal:
                principal
        """
        if principal > 0:
            if lender is not None and borrower is not None:
                try:
                    borrower.create_repos(lender, principal)
                except Exception as e:
                    print(e)
                    print("Strange! A Margin call failed at initialization.")
                    exit(1)
            else:
                if lender is not None:
                    repo1 = Repo(lender, None, principal)
                    lender.add(repo1)

                if borrower is not None:
                    repo2 = Repo(None, borrower, principal)
                    borrower.add(repo2)

    def initShares(self, owner, issuer, number):
        """adds shares to owner and issuer"""
        if issuer is None:
            return
        shares = issuer.issue_shares(owner, number)
        # owner.add(shares)  # AM Investor is disabled
        # issuer.add(shares)
        issuer.shares.append(shares)
        issuer.nShares_initial = issuer.nShares
        issuer.NAV_initial = issuer.get_net_asset_valuation()
        issuer.NAV_previous = issuer.NAV_initial
        assert issuer.NAV_initial > 0, issuer.NAV_initial

    def do_consistency_check(self, msg):
        # helper function for consistency check on accounting system
        print(msg)
        for agent in self.allAgents:
            if not agent.alive:
                continue
            A1 = agent.get_ledger().get_asset_valuation()
            L1 = agent.get_ledger().get_liability_valuation()
            E1 = A1 - L1
            A2 = agent.get_ledger().total_asset
            L2 = agent.get_ledger().total_liability
            E2 = A2 - L2
            delta = abs(E1 - E2) / E1
            assert delta <= 5e-2, (
                agent.get_name(),
                delta,
                E1,
                E2,
                A1 - A2,
                (A1 - A2) / A1,
                L1 - L2,
            )

    def do_sanity_check(self):
        for i in range(len(self.banks)):
            b = self.banks[i]
            ledger = b.get_ledger()
            A = ledger.get_asset_valuation()
            L = ledger.get_liability_valuation()
            eq = A - L
            data_A = self.snl_total_assets[i]
            data_L = self.snl_total_liabilities[i]
            data_eq = data_A - data_L
            diff_A = abs(data_A - A)
            A2 = (
                b.get_cash()
                + sum(
                    ledger.get_asset_valuation_of(_a)
                    for _a in [AssetCollateral, Loan, Repo, Other]
                )
                + ledger.get_asset_valuation_of(TradableAsset, AssetType.EXTERNAL1)
            )
            diff_A2 = abs(data_A - A2)
            diff_L = abs(data_L - L)
            # diff_CET1E = abs(b.get_CET1E(eq) - data_CET1E)
            diff_eq = abs(eq - data_eq)
            name = b.get_name()
            assert eq > 0, (name, eq)
            assert diff_A / data_A < 5e-15, (name, diff_A / data_A)
            assert diff_A2 / data_A < 5e-15, (name, diff_A2 / data_A)
            assert diff_L / data_L < 5e-15, (name, diff_L / data_L)
            assert diff_eq / data_eq < 8e-14, (name, diff_eq / data_eq, eq, data_eq)
            diff_lev = abs(self.snl_leverage[i] - b.get_leverage())
            assert diff_lev / self.snl_leverage[i] < 8e-14, (
                name,
                diff_lev / self.snl_leverage[i],
                b.DeltaE,
            )

    def load_bank_balancesheets_data(self):
        # read from data files
        if hasattr(self, "bank_balancesheets"):
            return
        # Taken from https://www.eba.europa.eu/risk-analysis-and-data/eu-wide-stress-testing/2016/results
        # https://resilience.zulipchat.com/#narrow/stream/110619-data-description/subject/words.201/near/121346521
        if DATA_YEAR == 2017:
            bs_file = "data/EBA_2018.csv"
            shock_file = "data/EBA_2018_shocks.csv"
        else:
            bs_file = "data/EBA_2016.csv"
            shock_file = "data/EBA_2016_shocks.csv"
        self.bank_balancesheets = read_csv(bs_file)
        # Prepare EBA shock to external asset scenario
        # first table page 3, sum corporates and retails,  defaulted A-IRB / (non-defaulted + defaulted)
        _eba_shocks_str = read_csv(shock_file)
        self.EBA_adverse_RWCR = [eval(i[0]) / 100 for i in _eba_shocks_str]
        self.EBA_adverse_lev = [eval(i[1]) / 100 for i in _eba_shocks_str]

    def setup_banks(self):
        banks = []
        self.load_bank_balancesheets_data()
        for i in range(NBANKS):
            row = self.bank_balancesheets[i]
            bank_name = row[0]
            if bank_name in self.banned_banks:
                # Skip banks with nan's
                continue
            bank = Bank(bank_name, self)
            banks.append(bank)
        self.setup_banks_tradable_assets_network(len(banks))
        return banks

    def setup_banks_tradable_assets_network(self, nbanks):
        # used only in setup_banks()
        if self.parameters.TRADABLE_ASSETS_ON and self.parameters.COMMON_ASSET_NETWORK:
            if self.govbonds_matrix is None and self.parameters.N_GOV_BONDS > 1:
                _seed = int(random.random() * 1000)
                self.govbonds_matrix, _ = generate_montagna_l3_network(
                    nbanks, self.parameters.N_GOV_BONDS, _seed
                )
                self.corpbonds_matrix, _ = generate_montagna_l3_network(
                    nbanks, self.parameters.N_CORP_BONDS, _seed + 1
                )
                self.equities_matrix, _ = generate_montagna_l3_network(
                    nbanks, self.parameters.N_EQUITIES, _seed + 2
                )
                self.othertradable_matrix, _ = generate_montagna_l3_network(
                    nbanks, self.parameters.N_OTHERTRADABLES, _seed + 3
                )
            # Add total common asset market the volume of
            # ((financial vehicle corps) + (insurance companies & pension funds) +
            #  (financial corporations engaging lending))

            # financial vehicle corp
            # http://sdw.ecb.europa.eu/servlet/desis?node=1000004041
            # cash -> deposit and loan claims, 2
            # equities -> 0
            # gov bonds -> 0.5 * (debt securities held)
            # corp bonds -> 0.5 * (debt securities held)
            # othertradable -> (securitized loans, 4) + (other securitized assets, 7) + (equity and investment fund shares/units, 8)
            # otherasset -> other assets, 9
            # 2017 Q4
            fhc_govbonds = 0.5 * 225100
            fhc_corpbonds = 0.5 * 225100
            fhc_equities = 0
            fhc_ot = 1222100 + 103400 + 72900

            # insurance companies & pension funds
            # http://sdw.ecb.europa.eu/reports.do?node=1000004038
            # cash -> currency and deposits
            # tradableequities -> shares and other equity
            # note: (securities other than shares) is gov+corp bonds
            # gov bonds -> (general goverment in table 2)
            # corp bonds -> (total in table 2) - (gov bonds)
            # othertradable -> (investment fund shares) + (money market fund shares)
            # otherasset -> (total assets) - (the rest)
            # 2016 Q2
            icpf_govbonds = 1935600
            icpf_corpbonds = 3931700 - 1935600
            icpf_equities = 984000
            icpf_ot = 2614000 + 116400

            # financial corporations engaging lending
            # http://sdw.ecb.europa.eu/servlet/desis?node=1000005697
            # tradableequities -> equity
            # gov bonds -> 0.5 * (debt securities held)
            # corp bonds -> 0.5 * (debt securities held)
            # othertradable -> 0
            # otherasset -> remaining assets
            # 2017 value (no quarter)
            fcel_govbonds = 0.5 * 32050
            fcel_corpbonds = 0.5 * 32050
            fcel_equities = 31374
            fcel_ot = 0

            # Used only in bail-in experiments where leveraged non-bank is on
            self.nonbank_govbonds = fhc_govbonds + icpf_govbonds + fcel_govbonds
            self.nonbank_corpbonds = fhc_corpbonds + icpf_corpbonds + fcel_corpbonds
            self.nonbank_equities = fhc_equities + icpf_equities + fcel_equities
            self.nonbank_ot = fhc_ot + icpf_ot + fcel_ot

            def sum_vecs(mat):
                # 4 lnb and 1 nlnb
                # renormalize by dividing by 5
                return (mat[-1] + mat[-2] + mat[-3] + mat[-4] + mat[-5]) / 5

            self.add_assetMarket_total_quantities(
                self.nonbank_govbonds * sum_vecs(self.govbonds_matrix),
                self.nonbank_corpbonds * sum_vecs(self.corpbonds_matrix),
                self.nonbank_equities * sum_vecs(self.equities_matrix),
                self.nonbank_ot * sum_vecs(self.othertradable_matrix),
            )
        else:
            raise Exception("TODO add other institutions volume to market volume")

    def setup_banks_generate_tradable_array(
        self, i, skipped, equities, gov_bonds, corp_bonds, other_tradable_assets
    ):
        equities_array = []
        gov_bonds_array = []
        corp_bonds_array = []
        other_tradable_array = []
        if self.parameters.TRADABLE_ASSETS_ON:
            if self.equities_matrix is not None:
                gov_bonds_array = gov_bonds * self.govbonds_matrix[i - skipped]
                corp_bonds_array = corp_bonds * self.corpbonds_matrix[i - skipped]
                equities_array = equities * self.equities_matrix[i]
                other_tradable_array = (
                    other_tradable_assets * self.othertradable_matrix[i - skipped]
                )
                self.add_assetMarket_total_quantities(
                    gov_bonds_array,
                    corp_bonds_array,
                    equities_array,
                    other_tradable_array,
                )
            else:
                equities_array = [equities]
                gov_bonds_array = [gov_bonds]
                corp_bonds_array = [corp_bonds]
                other_tradable_array = [other_tradable_assets]
                self.assetMarket.total_quantities[AssetType.EQUITIES1] += equities
                self.assetMarket.total_quantities[AssetType.GOV_BONDS1] += gov_bonds
                self.assetMarket.total_quantities[
                    AssetType.CORPORATE_BONDS1
                ] += corp_bonds
                self.assetMarket.total_quantities[
                    AssetType.OTHERTRADABLE1
                ] += other_tradable_assets
        return equities_array, gov_bonds_array, corp_bonds_array, other_tradable_array

    def setup_banks_balance_sheet_with_public_data(self):
        bank_counter = 0
        skipped = 0

        def _parse_row(row, i):
            return float(eval(row[i]))

        self.snl_cash = []
        self.snl_rrepo = []
        self.snl_interbank = []
        self.snl_repo = []
        self.snl_interbank_liability = []
        self.snl_CET1CR = []
        self.snl_LCR = []
        self.snl_total_assets = []
        self.snl_total_liabilities = []
        self.snl_leverage = []
        self.gov_bonds = []
        self.corp_bonds = []
        self.EBA_shocks_RWCR = []
        self.EBA_shocks_lev = []
        for i in range(NBANKS):
            row = self.bank_balancesheets[i]
            bank_name = row[0]
            if bank_name in self.banned_banks:
                # Skip banned banks
                skipped += 1
                continue
            rho_adv = self.EBA_adverse_RWCR[i]
            eba_govbonds = _parse_row(row, 1)
            eba_corpbonds = _parse_row(row, 2)
            eba_T2C = _parse_row(row, 5)
            eba_CET1E = _parse_row(row, 6)
            eba_leverage = _parse_row(row, 7) / 100
            eba_RWCR = _parse_row(row, 8) / 100
            eba_T1C = _parse_row(row, 9)
            AT1E = eba_T1C - eba_CET1E
            book_equity = eba_T1C
            DeltaE = book_equity - (eba_CET1E + AT1E + eba_T2C)
            assert AT1E >= 0
            A = eba_T1C / eba_leverage
            L = A - book_equity

            # used in do_sanity_check()
            self.snl_total_assets.append(A)
            self.snl_total_liabilities.append(L)
            self.snl_leverage.append(eba_leverage)
            self.snl_CET1CR.append(eba_RWCR)
            self.EBA_shocks_RWCR.append(rho_adv - eba_RWCR)
            self.EBA_shocks_lev.append(self.EBA_adverse_lev[i] - eba_leverage)
            self.gov_bonds.append(eba_govbonds)
            self.corp_bonds.append(eba_corpbonds)
            self.snl_LCR.append(120)
            cash = tradable = interbank_asset = rrepo = other_asset = external = A / 6
            # the prefix snl_ is used because initRepo(),
            # generate_montagna_l1_network(),
            # calibrate_LCR_weight_other(),
            # calibrate_RWA_weight(), expect those prefix as
            # input this
            self.snl_cash.append(cash)
            self.snl_rrepo.append(rrepo)
            self.snl_interbank.append(interbank_asset)
            eba_deposits = interbank_liability = repo = other_liability = L / 4
            self.snl_repo.append(repo)
            self.snl_interbank_liability.append(interbank_liability)
            tradable_bonds = eba_govbonds + eba_corpbonds
            tradable_nonbonds = tradable - tradable_bonds
            if tradable_nonbonds <= 0:
                tradable_nonbonds = tradable / 2
                # rescale govbonds and corpbonds so that their sum
                # is equal to tradable / 2
                eba_govbonds *= (tradable / 2) / tradable_bonds
                eba_corpbonds *= (tradable / 2) / tradable_bonds
            assert tradable_nonbonds >= 0, tradable_nonbonds
            (
                equities_array,
                gov_bonds_array,
                corp_bonds_array,
                other_tradable_array,
            ) = self.setup_banks_generate_tradable_array(
                i,
                skipped,
                tradable_nonbonds / 2,
                eba_govbonds,
                eba_corpbonds,
                tradable_nonbonds / 2,
            )
            _b = self.banks[bank_counter]
            _b.DeltaA = 0
            _b.AT1E = AT1E
            _b.T2C = eba_T2C
            _b.DeltaE = DeltaE
            _b.gamma4 = 0.5
            _b.gamma5 = 0.5
            _b.init(
                assets=(
                    cash,
                    equities_array,
                    corp_bonds_array,
                    gov_bonds_array,
                    other_tradable_array,
                    other_asset,
                ),
                liabilities=(0, 0),
            )
            self.add_other_liability(_b, other_liability)
            self.add_deposit(_b, eba_deposits)
            self.addExternalAsset(_b, AssetType.EXTERNAL1, external)
            bank_counter += 1

    def add_other_liability(self, bank, principal):
        bank.add(Other(None, bank, principal))

    def add_deposit(self, bank, principal):
        bank.add(Deposit(None, bank, principal))

    def setup_banks_interbank_network_from_loan_class(self, loan_class):
        # We parameterize this function with a loan class because the loan
        # class (BailinLoan) in the bailin model is different.
        # Generate Montagna-Kok 2016 interbank network
        if self.interbank_matrix is None:
            _seed = int(random.random() * 1000)
            np.random.seed(_seed)

            (
                self.interbank_matrix,
                self.interbank_liability_remainder,
            ) = generate_interbank_network_retry(
                self.snl_interbank, self.snl_interbank_liability
            )
        for i in range(len(self.banks)):
            for j in range(len(self.banks)):
                if i != j:
                    w = self.interbank_matrix[i, j]
                    if w != 0:
                        self.initInterBankLoan(self.banks[i], self.banks[j], w)

        # Init interbank loan to/from external node for the remainder
        for i in range(len(self.banks)):
            b = self.banks[i]
            w = self.interbank_liability_remainder[i]
            if w > 0:
                b.add(loan_class(None, b, w))
            elif w < 0:
                raise Exception(f"{b.get_name()}: w ({w}) must not be negative")

    def setup_banks_interbank_network(self):
        # This method is overridden in the bailin model to instead use
        # BailinLoan.
        self.setup_banks_interbank_network_from_loan_class(Loan)

    def setup_hedgefunds(self):
        hfs = []
        # See http://sdw.ecb.europa.eu/servlet/desis?node=1000003524
        # Discussion at https://resilience.zulipchat.com/#narrow/stream/src.2Fagents.2F/subject/HedgeFund/near/120017956
        # https://resilience.zulipchat.com/#narrow/stream/src.2Fnetwork/topic/Initialise.20Hedge.20Funds
        # Assets:
        # cash -> deposits and loan claims
        # tradable assets -> (debt securities) + equity
        # tradableequities -> equity
        # gov bonds -> debt securities * fraction_govbonds (split in same way as other investment funds )
        # corporate bonds -> debt securities * fraction_corpbonds (split in same way as other investment funds)
        # othertradable -> (investment fund shares) + (remaining assets and financial derivatives)
        # external -> 0
        # other asset -> (total asset) - (the rest of the assets)

        # Liabilities:
        # \tilde{R}_i, repo -> investment fund shares issued (this is deprecated)
        # E_i , equity -> data: loans and deposits, remaining liabilities and financial assets

        # Assume the fraction to be the same across representative AMs and
        # (OLD SOURCE) See http://sdw.ecb.europa.eu/servlet/desis?node=1000003528
        # Use data straight from HF table instead
        gov_bonds_fraction = (
            13900 / 120700
        )  # table 1.9.2.1 (General government) / (Total)
        corp_bonds_fraction = 1 - gov_bonds_fraction

        # Q = (aggregate HF assets size) / (total bank repo) * (1 - lambda_target)
        # Q4 2017
        aggregate_HF_asset = 490300
        lambda_target = self.parameters.HF_LEVERAGE_TARGET
        total_bank_rrepo = sum(self.snl_rrepo)
        Q = aggregate_HF_asset / total_bank_rrepo * (1 - lambda_target)

        cash_weight = 82700 / aggregate_HF_asset
        equities_weight = 93300 / aggregate_HF_asset
        debt_securities_weight = 120700 / aggregate_HF_asset
        gov_bonds_weight = gov_bonds_fraction * debt_securities_weight
        corp_bonds_weight = corp_bonds_fraction * debt_securities_weight
        othertradable_weight = (134700 + 58900) / aggregate_HF_asset
        otherAsset_weight = 1 - (
            cash_weight
            + equities_weight
            + debt_securities_weight
            + othertradable_weight
        )
        assert otherAsset_weight >= 0, otherAsset_weight

        nbanks = len(self.banks)
        for i in range(nbanks):
            total_asset = self.snl_rrepo[i] * Q / (1 - lambda_target)
            cash = cash_weight * total_asset
            equities = equities_weight * total_asset
            corp_bonds = corp_bonds_weight * total_asset
            gov_bonds = gov_bonds_weight * total_asset
            othertradable = othertradable_weight * total_asset
            otherAsset = otherAsset_weight * total_asset
            if self.equities_matrix is not None:
                equities_array = equities * self.equities_matrix[nbanks + i]
                corp_bonds_array = corp_bonds * self.corpbonds_matrix[nbanks + i]
                gov_bonds_array = gov_bonds * self.govbonds_matrix[nbanks + i]
                othertradable_array = (
                    othertradable * self.othertradable_matrix[nbanks + i]
                )
                self.add_assetMarket_total_quantities(
                    gov_bonds_array,
                    corp_bonds_array,
                    equities_array,
                    othertradable_array,
                )
                # Add tradable assets to non-bank's total quantities
                # Used only in bail-in experiments where leveraged non-bank is on
                self.nonbank_govbonds += gov_bonds_array
                self.nonbank_corpbonds += corp_bonds_array
                self.nonbank_equities += equities_array
                self.nonbank_ot += othertradable_array
            else:
                equities_array = [equities]
                gov_bonds_array = [0]
                corp_bonds_array = [corp_bonds]
                othertradable_array = [othertradable]
                self.assetMarket.total_quantities[AssetType.EQUITIES1] += equities
                self.assetMarket.total_quantities[AssetType.GOV_BONDS1] += gov_bonds
                self.assetMarket.total_quantities[
                    AssetType.CORPORATE_BONDS1
                ] += corp_bonds
                self.assetMarket.total_quantities[
                    AssetType.OTHERTRADABLE1
                ] += othertradable

            if self.parameters.HEDGEFUNDS_ON:
                hf = Hedgefund("%s" % self.banks[i].get_name(), self)
                hf.init(
                    assets=(
                        cash,
                        equities_array,
                        corp_bonds_array,
                        gov_bonds_array,
                        othertradable_array,
                        otherAsset,
                    ),
                    liabilities=(0, 0),
                )
                if self.parameters.HF_USE_FUNDAMENTALIST_STRATEGY:
                    hf.equities_fundamental = np.array(equities_array)
                    hf.corp_bonds_fundamental = np.array(corp_bonds_array)
                    hf.gov_bonds_fundamental = np.array(gov_bonds_array)
                    hf.othertradable_fundamental = np.array(othertradable_array)
                    hf.other_fundamental = otherAsset
                hfs.append(hf)
        return hfs, Q

    def setup_asset_managers(self):
        # Initialise 5 representative asset managers
        # Representative Bond Fund, Representative Equity Fund, Representative Mixed Fund, Representative Real Estate Fund, Representative Other Fund
        # See https://resilience.zulipchat.com/#narrow/stream/src.2Fagents.2F/subject/AssetManager/near/120016466 for how to read the warehouse data
        # outstanding amounts instead of transactions
        # Asset:
        # cash -> deposits and loan claims
        # tradableequities -> equity
        # corp+gov bonds -> debt securities
        # gov bonds -> (table 2, general government)
        # corp bonds -> (debt securities) - (gov bonds)
        # othertradable -> (investment fund shares) + (remaining assets and financial derivatives)
        # external -> 0
        # other asset -> (total asset) - (the rest of the assets)

        # Liability:
        # otherLiability: (remaining liabilities and financial derivatives) + (loans and deposits received)
        # asset manager shares (equity): Investment fund shares issued
        if self.parameters.ASSET_MANAGER_ON:
            bond_am = AssetManager("Rep. Bond Fund", self)
            equity_am = AssetManager("Rep. Equity Fund", self)
            mixed_am = AssetManager("Rep. Mixed Fund", self)
            realestate_am = AssetManager("Rep. Real Estate Fund", self)
            other_am = AssetManager("Rep. Other Fund", self)
        else:
            bond_am = None
            equity_am = None
            mixed_am = None
            realestate_am = None
            other_am = None

        # Disabled
        # inv1 = Investor("Investor 1", self.simulation)
        # inv1.init(assets=(0, [], [], [], [], 0), liabilities=(0, 0, 0))
        # invs = [inv1]
        inv1 = None
        invs = []

        # constraint: asset manager's liability (the sum of the last 3 params of init)
        # is \tilde{O}_i, see equation 7
        # data source for asset manager: http://sdw.ecb.europa.eu/reports.do?node=1000003515
        am_offset = len(self.banks) * 2

        def _init_am(
            am,
            offset,
            cash,
            equities,
            debt_securities,
            gb,
            ot,
            total_asset,
            otherLiability,
            equity,
        ):
            equities_array = equities * self.equities_matrix[am_offset + offset]
            corp_bonds_array = (debt_securities - gb) * self.corpbonds_matrix[
                am_offset + offset
            ]
            gov_bonds_array = gb * self.govbonds_matrix[am_offset + offset]
            othertradable_array = ot * self.othertradable_matrix[am_offset + offset]
            other_asset = total_asset - (cash + equities + debt_securities + ot)
            assert other_asset >= 0, other_asset

            if self.parameters.ASSET_MANAGER_ON:
                am.init(
                    assets=(
                        cash,
                        equities_array,
                        corp_bonds_array,
                        gov_bonds_array,
                        othertradable_array,
                        other_asset,
                    ),
                    liabilities=(0, 0),
                )
                am.add(Other(None, am, otherLiability))
                self.initShares(inv1, am, 100)  # note: this has to be done last!
                am.cash_fraction_initial = (
                    am.get_cash() / am.get_ledger().get_asset_valuation()
                )
                assert isequal_float(am.get_equity_valuation(), equity, 601), (
                    am.get_equity_valuation(),
                    equity,
                )

            # Add tradable assets to asset market's total quantities
            # Used only in bail-in experiments where leveraged non-bank is on
            self.add_assetMarket_total_quantities(
                gov_bonds_array, corp_bonds_array, equities_array, othertradable_array
            )
            # Add tradable assets to non-bank's total quantities
            self.nonbank_govbonds += gov_bonds_array
            self.nonbank_corpbonds += corp_bonds_array
            self.nonbank_equities += equities_array
            self.nonbank_ot += othertradable_array

        # See http://sdw.ecb.europa.eu/servlet/desis?node=1000003519
        # Q4 2017
        _init_am(
            bond_am,
            0,
            cash=175800,
            equities=32400,
            debt_securities=2861000,
            gb=526800,
            ot=219800 + 229300,
            total_asset=3518300,
            otherLiability=233900 + 48000,
            equity=3236500,
        )

        # See http://sdw.ecb.europa.eu/servlet/desis?node=1000003520
        # Q4 2017
        _init_am(
            equity_am,
            1,
            cash=100300,
            equities=2850800,
            debt_securities=53200,
            gb=9300,
            ot=276000 + 141100,
            total_asset=3422200,
            otherLiability=136400 + 25100,
            equity=3261000,
        )

        # See http://sdw.ecb.europa.eu/servlet/desis?node=1000003521
        # Q4 2017
        _init_am(
            mixed_am,
            2,
            cash=180100,
            equities=551500,
            debt_securities=1173700,
            gb=264600,
            ot=1016400 + 185900,
            total_asset=3109300,
            otherLiability=165100 + 42200,
            equity=2902100,
        )

        # See http://sdw.ecb.europa.eu/servlet/desis?node=1000003522
        # Q4 2017
        _init_am(
            realestate_am,
            3,
            cash=101100,
            equities=112800,
            debt_securities=11700,
            gb=1000,
            ot=68500 + 77100,
            total_asset=722800,
            otherLiability=41100 + 106300,
            equity=575500,
        )

        # See http://sdw.ecb.europa.eu/servlet/desis?node=1000003526
        # Q4 2017
        _init_am(
            other_am,
            4,
            cash=177500,
            equities=184800,
            debt_securities=286900,
            gb=38100,
            ot=323400 + 88200,
            total_asset=1070500,
            otherLiability=66700 + 129900,
            equity=874500,
        )

        ams = (
            [bond_am, equity_am, mixed_am, realestate_am, other_am]
            if self.parameters.ASSET_MANAGER_ON
            else []
        )
        return ams, invs

    def calibrate_RWA_weight(self):
        for i in range(len(self.banks)):
            b = self.banks[i]
            b.RWA_weights = dict(self.parameters.RWA_WEIGHTS_GROUPED)
            CET1E = b.get_CET1E()
            snl_rwa = CET1E / self.snl_CET1CR[i]
            rwa = b.rwa_constraint.get_RWA()

            rw_other = snl_rwa - rwa
            do_mul = True
            if rw_other >= 0:
                # Only match the RWA data when risk-weighted other asset is positive
                other_asset = b.get_ledger().get_asset_valuation_of(Other)
                if other_asset > 0:
                    weight = rw_other / other_asset
                    if weight < 1:
                        b.RWA_weights["other"] = weight
                        do_mul = False
                    else:
                        b.RWA_weights["other"] = 1
                else:
                    # other asset is 0
                    b.RWA_weights["other"] = 1

            if do_mul:
                # Rescale the risk-weights except risk-weight-other so that it
                # matches the data
                mul = snl_rwa / rwa
                for k, v in b.RWA_weights.items():
                    if k == "other":
                        # already calibrated
                        continue
                    b.RWA_weights[k] = mul * v

            b.LCR_weight_other = 0
            b.LEVERAGE_INITIAL = b.get_leverage()

    def calibrate_LCR_weight_other(self):
        # Filling in the nan values in LCR array with LCR_average
        # Calculating "Other LCR coefficient"
        LCR_average = np.nanmean(self.snl_LCR)
        self.snl_LCR = [LCR_average if np.isnan(i) else i for i in self.snl_LCR]
        BASELIII_CASH_OUTFLOW_CAP = 0.75
        predicted_weights = [
            1.8,
            0.8,
            1,
            0.5,
            0.5,
            0.5,
            1,
            1,
            0.9,
            1,
            1,
            1,
            0.5,
            1,
            1,
            0.5,
            0.4,
            0.4,
            5.4,
            4.6,
            1,
            0.45,
            1.2,
            1,
            1.4,
            1,
            0.7,
            3.7,
            1,
            1,
            1,
            1,
            0.5,
            0.7,
            0.8,
            0.8,
            1,
            1,
            1,
            1,
        ]
        for i in range(len(self.banks)):
            b = self.banks[i]
            LCR = self.snl_LCR[i]
            LCR_numerator1 = self.snl_cash[i] + self.gov_bonds[i]
            # denominator
            den = LCR_numerator1 / (LCR / 100)
            b.LCR_den_initial = den
            continue
            b.LCR_weight_other = 0
            inflows = b.lcr_constraint.get_inflows()
            outflows = b.lcr_constraint.get_outflows()
            other_lia = self.snl_other_liability[i]

            expected_inflows_cap = BASELIII_CASH_OUTFLOW_CAP * (
                outflows + other_lia * predicted_weights[i]
            )
            if expected_inflows_cap > inflows:
                o_i = outflows - inflows
                other_lcrw = (den - o_i) / other_lia
            else:
                other_lcrw = (
                    den / (1 - BASELIII_CASH_OUTFLOW_CAP) - outflows
                ) / other_lia
            if other_lcrw < 0:
                other_lcrw = 0
            b.LCR_weight_other = other_lcrw
            _new_outflows = outflows + other_lcrw * other_lia
            if other_lcrw > 0:
                den_computed = _new_outflows - min(
                    inflows, BASELIII_CASH_OUTFLOW_CAP * _new_outflows
                )
                assert_equal(den, den_computed)

    def apply_initial_shock(self, assetType, fraction):
        """creates an initial shock, by decreasing the prices on the asset market"""
        self.assetMarket.set_price(
            assetType, self.assetMarket.get_price(assetType) * (1.0 - fraction)
        )

        for agent in self.allAgents:
            agent.receive_shock_to_asset(assetType, fraction)

    def devalueCommonAsset(self, assetType, priceLost):
        """devaluates a common asset for all agents"""
        for agent in self.allAgents:
            agent.devalue_asset_collateral_of_type(assetType, priceLost)

    def check_for_alive(self):
        return [b.is_alive() for b in self.banks]

    def add_assetMarket_total_quantities(self, gb_arr, cb_arr, eq_arr, ot_arr):
        # gov_bonds is numbered from 1 to N_GOV_BONDS
        _N = 0
        for _i in range(1, self.parameters.N_GOV_BONDS + 1):
            self.assetMarket.total_quantities[_N + _i] += gb_arr[_i - 1]
        # corp_bonds is numbered from N_GOV_BONDS + 1 to N_GOV_BONDS + N_CORP_BONDS
        _N += self.parameters.N_GOV_BONDS
        for _i in range(1, self.parameters.N_CORP_BONDS + 1):
            self.assetMarket.total_quantities[_N + _i] += cb_arr[_i - 1]
        # equities is numbered from N_GOV_BONDS + N_CORP_BONDS + 1 to N_GOV_BONDS + N_CORP_BONDS + N_EQUITIES + 1
        _N += self.parameters.N_CORP_BONDS
        for _i in range(1, self.parameters.N_EQUITIES + 1):
            self.assetMarket.total_quantities[_N + _i] += eq_arr[_i - 1]
        # othertradable is numbered from N_GOV_BONDS + N_CORP_BONDS + N_EQUITIES + 1 to N_GOV_BONDS + N_CORP_BONDS + N_EQUITIES + N_OTHERTRADABLES + 1
        _N += self.parameters.N_EQUITIES
        for _i in range(1, self.parameters.N_OTHERTRADABLES + 1):
            self.assetMarket.total_quantities[_N + _i] += ot_arr[_i - 1]

    def reset_networks(self):
        self.interbank_matrix = None
        self.govbonds_matrix = None
        self.corpbonds_matrix = None
        self.equities_matrix = None
        self.othertradable_matrix = None
