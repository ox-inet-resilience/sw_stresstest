from resilience.parameters import (  # noqa
    Parameters as ResilienceParameters,
    enum,
    isequal_float,
    eps,
)

DATA_YEAR = 2017
# DATA_YEAR = 2015

GSIBs_2015 = [
    "UK48",
    "UK49",
    "DE15",
    "FR13",
    "FR12",
    "NL31",
    "SE44",
    "UK50",
    "ES38",
    "IT28",
]
GSIBs_2017 = ["UK45", "UK46", "DE17", "FR12", "NL32", "SE41", "UK48", "ES39", "IT28"]
GSIBs = GSIBs_2017 if DATA_YEAR == 2017 else GSIBs_2015
GSIBs_name = {  # 2017
    "UK45": "Barclays",  # Barclays Plc
    "UK46": "HSBC",  # HSBC Holdings Plc
    "DE17": "Deutsche",  # Deutsche Bank AG
    "FR12": "Agricole",  # Crédit Agricole SA
    "NL32": "ING",  # ING Groep N.V.
    "SE41": "Nordea",  # Nordea Bank AB
    "UK48": "RBS",  # Royal Bank of Scotland Group Plc
    "ES39": "Santander",  # Banco Santander
    "IT28": "UniCredit",  # UniCredit SpA
}

N_GOV_BONDS = 21
N_CORP_BONDS = 24
N_EQUITIES = 24
N_OTHERTRADABLES = 24
_offset = 0
govbonds_dict = {f"GOV_BONDS{i}": (_offset + i) for i in range(1, N_GOV_BONDS + 1)}
_offset += N_GOV_BONDS
corpbonds_dict = {
    f"CORPORATE_BONDS{i}": (_offset + i) for i in range(1, N_CORP_BONDS + 1)
}
_offset += N_CORP_BONDS
equities_dict = {f"EQUITIES{i}": (_offset + i) for i in range(1, N_EQUITIES + 1)}
_offset += N_EQUITIES
othertradables_dict = {
    f"OTHERTRADABLE{i}": (_offset + i) for i in range(1, N_OTHERTRADABLES + 1)
}

AssetType = enum(
    **{
        **govbonds_dict,
        **corpbonds_dict,
        **equities_dict,
        **othertradables_dict,
        **dict(EXTERNAL1=422, EXTERNAL2=423, EXTERNAL3=424),
    }
)


class Parameters(ResilienceParameters):
    # Initial adverse scenario parameters -----
    SYSTEMWIDE_SHOCK_SCALE = 1  # Adverse shock relative to 2018 EBA scenario
    ASSET_TO_SHOCK = AssetType.EXTERNAL1

    # These params control what type of y-axis to be plotted
    # If on, asset loss is plotted on y axis instead of bank defaults.
    PLOT_ASSET_LOSS = False

    # Simulation time steps
    # 6 timesteps are used for the foundation paper;
    # 12 for bail-in paper
    SIMULATION_TIMESTEPS = 6

    # Agents on or off
    ASSET_MANAGER_ON = False
    HEDGEFUNDS_ON = False

    # Bank regulatory constraints
    BANK_RWA_ON = True
    BANK_LEVERAGE_ON = False
    BANK_LCR_ON = False

    # Target above buffer value for regulatory constraints
    BANK_RWA_EXCESS_TARGET = 0.01  # foundation settings DO NOT MODIFY!!
    RHO_M_STANDARD = 0.045
    # BANK_LEVERAGE_EXCESS_TARGET and BANK_LCR_EXCESS_TARGET are
    # defined in resilience

    # Buffer value at which point banks act to return to target
    # (usability of buffers)
    # TODO use this
    # BANK_RWCR_USABILITY_BUFFER = 0.5  # where rho_buffer = rho_M + BANK_RWCR_USABILITY_BUFFER * rho_CB
    # BANK_LR_USABILITY_BUFFER = 0.5
    # BANK_LCR_USABILITY_BUFFER = 0.5

    # Miscellaneous (e.g. asset market; price impact; GSIB banks;
    # RW in RWAs) -----
    GSIBs = GSIBs
    N_GOV_BONDS = N_GOV_BONDS
    N_CORP_BONDS = N_CORP_BONDS
    N_EQUITIES = N_EQUITIES
    N_OTHERTRADABLES = N_OTHERTRADABLES
    AssetType = AssetType

    # Price impacts
    def price_impact_fn_pi(pi):
        sold = 0.05
        # exponential
        # beta = -1 / sold * np.log(1 - pi)
        # linear
        # the price drops by `pi` amount if `sold` amount of the market cap has been sold
        beta = pi / sold

        return {
            AssetType.EXTERNAL1: beta,
            **{
                k: beta / 2 for k in govbonds_dict.values()
            },  # GOV_BONDS is always set to have 0 impact
            **{k: beta for k in corpbonds_dict.values()},  # CORP_BONDS
            **{
                k: 2 * beta for k in equities_dict.values()
            },  # EQUITIES has twice the price impact of the rest
            **{k: beta for k in othertradables_dict.values()},
        }  # OTHERTRADABLES

    PRICE_IMPACTS = None  # make sure to always generate this dict!!

    # According to p67
    INITIAL_HAIRCUTS = {
        **{k: 0.04 for k in corpbonds_dict.values()},
        **{k: 0.02 for k in govbonds_dict.values()},
        **{k: 0.15 for k in equities_dict.values()},
        **{k: 0.04 for k in othertradables_dict.values()},  # the same as corpbonds
    }

    RWA_WEIGHTS_GROUPED = {
        "corpbonds": 1.00,
        "govbonds": 0.00,
        "equities": 0.75,
        "othertradables": 1.00,  # same as corpbonds
        "loan": 0.4,
        "repo": 0.1,
        "external": 0.35,
        # this is a regularization so that RWA never goes to 0. sometimes
        # external asset is 0 which prevents it from being a stopping point for
        # the RWA to go to 0
        "other": 0.01,
    }

    govbonds_dict = govbonds_dict
    corpbonds_dict = corpbonds_dict
    equities_dict = equities_dict
    othertradables_dict = othertradables_dict
