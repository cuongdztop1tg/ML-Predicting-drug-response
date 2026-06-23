from __future__ import annotations

import json
import math
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


RANDOM_STATE = 42
TARGET_COL = "auc"
CELL_ID_COL = "improve_sample_id"
DRUG_ID_COL = "improve_chem_id"
BENCHMARK_DATASETS = ["gCSI", "CCLE", "GDSCv2", "GDSCv1", "CTRPv2"]
BENCHMARK_FOLDS = list(range(10))
DEFAULT_MODELS = ["ridge", "random_forest", "lightgbm", "graphdrp", "simple_linear_nn"]
# L1000 landmark genes from Subramanian et al. 2017, Table S2; control INV probes excluded.
# Keeping this local makes Kaggle runs independent of JDACS4C-IMPROVE/improvelib.statics.
LINCS_SYMBOL_SOURCE = "L1000/Subramanian 2017 Table S2, 976 landmark genes"
LINCS_SYMBOL = [
    'AARS', 'ABCB6', 'ABCC5', 'ABCF1', 'ABCF3', 'ABHD4', 'ABHD6', 'ABL1',
    'ACAA1', 'ACAT2', 'ACBD3', 'ACD', 'ACLY', 'ACOT9', 'ADAM10', 'ADAT1',
    'ADGRE5', 'ADGRG1', 'ADH5', 'ADI1', 'ADO', 'ADRB2', 'AGL', 'AKAP8',
    'AKAP8L', 'AKR7A2', 'AKT1', 'ALAS1', 'ALDH7A1', 'ALDOA', 'ALDOC', 'AMDHD2',
    'ANKRD10', 'ANO10', 'ANXA7', 'APBB2', 'APOE', 'APP', 'APPBP2', 'ARFIP2',
    'ARHGAP1', 'ARHGEF12', 'ARHGEF2', 'ARID4B', 'ARID5B', 'ARL4C', 'ARNT2', 'ARPP19',
    'ASAH1', 'ASCC3', 'ATF1', 'ATF5', 'ATF6', 'ATG3', 'ATMIN', 'ATP11B',
    'ATP1B1', 'ATP2C1', 'ATP5S', 'ATP6V0B', 'ATP6V1D', 'AURKA', 'AURKB', 'AXIN1',
    'B4GAT1', 'BACE2', 'BAD', 'BAG3', 'BAMBI', 'BAX', 'BCL2', 'BCL7B',
    'BDH1', 'BECN1', 'BHLHE40', 'BID', 'BIRC2', 'BIRC5', 'BLCAP', 'BLMH',
    'BLVRA', 'BMP4', 'BNIP3', 'BNIP3L', 'BPHL', 'BRCA1', 'BTK', 'BUB1B',
    'BZW2', 'C2CD2', 'C2CD2L', 'C2CD5', 'C5', 'CAB39', 'CALM3', 'CALU',
    'CAMSAP2', 'CANT1', 'CAPN1', 'CARMIL1', 'CASC3', 'CASK', 'CASP10', 'CASP2',
    'CASP3', 'CASP7', 'CAST', 'CAT', 'CBLB', 'CBR1', 'CBR3', 'CCDC85B',
    'CCDC86', 'CCDC92', 'CCL2', 'CCNA1', 'CCNA2', 'CCNB1', 'CCNB2', 'CCND1',
    'CCND3', 'CCNE2', 'CCNF', 'CCNH', 'CCP110', 'CD320', 'CD40', 'CD44',
    'CD58', 'CDC20', 'CDC25A', 'CDC25B', 'CDC42', 'CDC45', 'CDCA4', 'CDH3',
    'CDK1', 'CDK19', 'CDK2', 'CDK4', 'CDK5R1', 'CDK6', 'CDK7', 'CDKN1A',
    'CDKN1B', 'CDKN2A', 'CEBPA', 'CEBPD', 'CEBPZ', 'CENPE', 'CEP57', 'CERK',
    'CETN3', 'CFLAR', 'CGRRF1', 'CHAC1', 'CHEK1', 'CHEK2', 'CHERP', 'CHIC2',
    'CHMP4A', 'CHMP6', 'CHN1', 'CHP1', 'CIAPIN1', 'CIRBP', 'CISD1', 'CLIC4',
    'CLPX', 'CLSTN1', 'CLTB', 'CLTC', 'CNDP2', 'CNOT4', 'CNPY3', 'COASY',
    'COG2', 'COG4', 'COG7', 'COL1A1', 'COL4A1', 'COPB2', 'COPS7A', 'COQ8A',
    'CORO1A', 'CPNE3', 'CPSF4', 'CREB1', 'CREG1', 'CRELD2', 'CRK', 'CRKL',
    'CRTAP', 'CRYZ', 'CSK', 'CSNK1A1', 'CSNK1E', 'CSNK2A2', 'CSRP1', 'CTNNAL1',
    'CTNND1', 'CTSD', 'CTSL', 'CTTN', 'CXCL2', 'CXCR4', 'CYB561', 'CYCS',
    'CYTH1', 'DAG1', 'DAXX', 'DCK', 'DCTD', 'DCUN1D4', 'DDB2', 'DDIT4',
    'DDR1', 'DDX10', 'DDX42', 'DECR1', 'DENND2D', 'DERA', 'DFFA', 'DFFB',
    'DHDDS', 'DHRS7', 'DHX29', 'DLD', 'DMTF1', 'DNAJA3', 'DNAJB1', 'DNAJB2',
    'DNAJB6', 'DNAJC15', 'DNM1', 'DNM1L', 'DNMT1', 'DNMT3A', 'DNTTIP2', 'DPH2',
    'DRAP1', 'DSG2', 'DUSP11', 'DUSP14', 'DUSP22', 'DUSP3', 'DUSP4', 'DUSP6',
    'DYNLT3', 'DYRK3', 'E2F2', 'EAPP', 'EBNA1BP2', 'EBP', 'ECD', 'ECH1',
    'EDEM1', 'EDN1', 'EED', 'EFCAB14', 'EGF', 'EGFR', 'EGR1', 'EIF4EBP1',
    'EIF4G1', 'EIF5', 'ELAC2', 'ELAVL1', 'ELOVL6', 'ELP1', 'EML3', 'ENOPH1',
    'ENOSF1', 'EPB41L2', 'EPHA3', 'EPHB2', 'EPN2', 'EPRS', 'ERBB2', 'ERBB3',
    'ERO1A', 'ETFB', 'ETS1', 'ETV1', 'EVL', 'EXOSC4', 'EXT1', 'EZH2',
    'FAH', 'FAIM', 'FAM20B', 'FAM57A', 'FAM69A', 'FAS', 'FASTKD5', 'FAT1',
    'FBXL12', 'FBXO11', 'FBXO21', 'FBXO7', 'FCHO1', 'FDFT1', 'FEZ2', 'FGFR2',
    'FGFR4', 'FHL2', 'FIS1', 'FKBP14', 'FKBP4', 'FOS', 'FOSL1', 'FOXJ3',
    'FOXO3', 'FOXO4', 'FPGS', 'FRS2', 'FSD1', 'FUT1', 'FYN', 'FZD1',
    'FZD7', 'G3BP1', 'GAA', 'GABPB1', 'GADD45A', 'GADD45B', 'GALE', 'GAPDH',
    'GATA2', 'GATA3', 'GDPD5', 'GFOD1', 'GFPT1', 'GHR', 'GLI2', 'GLOD4',
    'GLRX', 'GMNN', 'GNA11', 'GNA15', 'GNAI1', 'GNAI2', 'GNAS', 'GNB5',
    'GNPDA1', 'GOLT1B', 'GPATCH8', 'GPC1', 'GPER1', 'GRB10', 'GRB7', 'GRN',
    'GRWD1', 'GSTM2', 'GSTZ1', 'GTF2A2', 'GTF2E2', 'GTPBP8', 'H2AFV', 'HACD3',
    'HADH', 'HAT1', 'HDAC2', 'HDAC6', 'HDGFL3', 'HEATR1', 'HEBP1', 'HERC6',
    'HERPUD1', 'HES1', 'HIF1A', 'HIST1H2BK', 'HIST2H2BE', 'HK1', 'HLA-DMA', 'HLA-DRA',
    'HMG20B', 'HMGA2', 'HMGCR', 'HMGCS1', 'HMOX1', 'HOMER2', 'HOOK2', 'HOXA10',
    'HOXA5', 'HPRT1', 'HS2ST1', 'HSD17B10', 'HSD17B11', 'HSPA1A', 'HSPA4', 'HSPA8',
    'HSPB1', 'HSPD1', 'HTATSF1', 'HTRA1', 'HYOU1', 'IARS2', 'ICAM1', 'ICAM3',
    'ICMT', 'ID2', 'IDE', 'IER3', 'IFNAR1', 'IFRD2', 'IGF1R', 'IGF2BP2',
    'IGF2R', 'IGFBP3', 'IGHMBP2', 'IKBKB', 'IKBKE', 'IKZF1', 'IL13RA1', 'IL1B',
    'IL4R', 'ILK', 'INPP1', 'INPP4B', 'INSIG1', 'INTS3', 'IPO13', 'IQGAP1',
    'ISOC1', 'ITFG1', 'ITGAE', 'ITGB1BP1', 'ITGB5', 'JADE2', 'JMJD6', 'JPT2',
    'JUN', 'KAT6A', 'KAT6B', 'KCNK1', 'KCTD5', 'KDELR2', 'KDM3A', 'KDM5A',
    'KDM5B', 'KEAP1', 'KIAA0100', 'KIAA0355', 'KIAA0753', 'KIAA0907', 'KIF14', 'KIF1BP',
    'KIF20A', 'KIF2C', 'KIF5C', 'KIT', 'KLHDC2', 'KLHL21', 'KLHL9', 'KTN1',
    'LAGE3', 'LAMA3', 'LAP3', 'LBR', 'LGALS8', 'LGMN', 'LIG1', 'LIPA',
    'LOXL1', 'LPAR2', 'LPGAT1', 'LRP10', 'LRPAP1', 'LRRC41', 'LSM5', 'LSM6',
    'LSR', 'LYN', 'LYPLA1', 'LYRM1', 'MACF1', 'MALT1', 'MAMLD1', 'MAN2B1',
    'MAP2K5', 'MAP3K4', 'MAP4K4', 'MAP7', 'MAPK13', 'MAPK1IP1L', 'MAPK9', 'MAPKAPK2',
    'MAPKAPK3', 'MAPKAPK5', 'MAST2', 'MAT2A', 'MBNL1', 'MBNL2', 'MBOAT7', 'MBTPS1',
    'MCM3', 'MCOLN1', 'MCUR1', 'ME2', 'MEF2C', 'MELK', 'MEST', 'METRN',
    'MFSD10', 'MICALL1', 'MIF', 'MINDY1', 'MKNK1', 'MLEC', 'MLLT11', 'MMP1',
    'MMP2', 'MNAT1', 'MOK', 'MPC2', 'MPZL1', 'MRPL12', 'MRPL19', 'MRPS16',
    'MRPS2', 'MSH6', 'MSRA', 'MTA1', 'MTERF3', 'MTF2', 'MTFR1', 'MTHFD2',
    'MUC1', 'MVP', 'MYBL2', 'MYC', 'MYCBP', 'MYCBP2', 'MYL9', 'MYLK',
    'NARFL', 'NCAPD2', 'NCK1', 'NCK2', 'NCOA3', 'NENF', 'NET1', 'NFATC3',
    'NFATC4', 'NFE2L2', 'NFIL3', 'NFKB2', 'NFKBIA', 'NFKBIB', 'NFKBIE', 'NGRN',
    'NIPSNAP1', 'NISCH', 'NIT1', 'NMT1', 'NNT', 'NOL3', 'NOLC1', 'NOS3',
    'NOSIP', 'NOTCH1', 'NPC1', 'NPDC1', 'NPEPL1', 'NPRL2', 'NR1H2', 'NR2F6',
    'NR3C1', 'NRAS', 'NRIP1', 'NSDHL', 'NT5DC2', 'NUCB2', 'NUDCD3', 'NUDT9',
    'NUP133', 'NUP62', 'NUP85', 'NUP88', 'NUP93', 'NUSAP1', 'NVL', 'ORC1',
    'OXA1L', 'OXCT1', 'OXSR1', 'P4HA2', 'P4HTM', 'PACSIN3', 'PAF1', 'PAFAH1B1',
    'PAFAH1B3', 'PAICS', 'PAK1', 'PAK4', 'PAK6', 'PAN2', 'PAPD7', 'PARP1',
    'PARP2', 'PAX8', 'PCBD1', 'PCCB', 'PCK2', 'PCM1', 'PCMT1', 'PCNA',
    'PDGFA', 'PDHX', 'PDIA5', 'PDLIM1', 'PDS5A', 'PECR', 'PEX11A', 'PFKL',
    'PGAM1', 'PGM1', 'PGRMC1', 'PHGDH', 'PHKA1', 'PHKB', 'PHKG2', 'PIGB',
    'PIH1D1', 'PIK3C2B', 'PIK3C3', 'PIK3CA', 'PIK3R3', 'PIK3R4', 'PIN1', 'PIP4K2B',
    'PKIG', 'PLA2G15', 'PLA2G4A', 'PLCB3', 'PLEKHJ1', 'PLEKHM1', 'PLK1', 'PLOD3',
    'PLP2', 'PLS1', 'PLSCR1', 'PLSCR3', 'PMAIP1', 'PMM2', 'PNKP', 'PNP',
    'POLB', 'POLD4', 'POLE2', 'POLG2', 'POLR1C', 'POLR2I', 'POLR2K', 'PPARD',
    'PPARG', 'PPIC', 'PPIE', 'PPOX', 'PPP1R13B', 'PPP2R3C', 'PPP2R5A', 'PPP2R5E',
    'PRAF2', 'PRCP', 'PRKACA', 'PRKAG2', 'PRKCD', 'PRKCH', 'PRKCQ', 'PRKX',
    'PROS1', 'PRPF4', 'PRR15L', 'PRR7', 'PRSS23', 'PRUNE1', 'PSIP1', 'PSMB10',
    'PSMB8', 'PSMD10', 'PSMD2', 'PSMD4', 'PSMD9', 'PSME1', 'PSME2', 'PSMF1',
    'PSMG1', 'PSRC1', 'PTGS2', 'PTK2', 'PTK2B', 'PTPN1', 'PTPN12', 'PTPN6',
    'PTPRC', 'PTPRF', 'PTPRK', 'PUF60', 'PWP1', 'PXMP2', 'PXN', 'PYCR1',
    'PYGL', 'RAB11FIP2', 'RAB21', 'RAB27A', 'RAB31', 'RAB4A', 'RAC2', 'RAD51C',
    'RAD9A', 'RAE1', 'RAI14', 'RALA', 'RALB', 'RALGDS', 'RAP1GAP', 'RASA1',
    'RB1', 'RBKS', 'RBM15B', 'RBM34', 'RBM6', 'REEP5', 'RELB', 'RFC2',
    'RFC5', 'RFNG', 'RFX5', 'RGS2', 'RHEB', 'RHOA', 'RNF167', 'RNH1',
    'RNMT', 'RNPS1', 'RPA1', 'RPA2', 'RPA3', 'RPIA', 'RPL39L', 'RPN1',
    'RPP38', 'RPS5', 'RPS6', 'RPS6KA1', 'RRAGA', 'RRP12', 'RRP1B', 'RRP8',
    'RRS1', 'RSU1', 'RTN2', 'RUVBL1', 'S100A13', 'S100A4', 'SACM1L', 'SATB1',
    'SCAND1', 'SCARB1', 'SCCPDH', 'SCP2', 'SCRN1', 'SCYL3', 'SDHB', 'SENP6',
    'SERPINE1', 'SESN1', 'SFN', 'SGCB', 'SH3BP5', 'SHB', 'SHC1', 'SIRT3',
    'SKIV2L', 'SKP1', 'SLC11A2', 'SLC1A4', 'SLC25A13', 'SLC25A14', 'SLC25A4', 'SLC25A46',
    'SLC27A3', 'SLC2A6', 'SLC35A1', 'SLC35A3', 'SLC35B1', 'SLC35F2', 'SLC37A4', 'SLC5A6',
    'SMAD3', 'SMARCA4', 'SMARCC1', 'SMARCD2', 'SMC1A', 'SMC3', 'SMC4', 'SMNDC1',
    'SNAP25', 'SNCA', 'SNX11', 'SNX13', 'SNX6', 'SNX7', 'SOCS2', 'SORBS3',
    'SOX2', 'SOX4', 'SPAG4', 'SPAG7', 'SPDEF', 'SPEN', 'SPP1', 'SPR',
    'SPRED2', 'SPTAN1', 'SPTLC2', 'SQOR', 'SQSTM1', 'SRC', 'SSBP2', 'ST3GAL5',
    'ST6GALNAC2', 'ST7', 'STAMBP', 'STAP2', 'STAT1', 'STAT3', 'STAT5B', 'STK10',
    'STK25', 'STMN1', 'STUB1', 'STX1A', 'STX4', 'STXBP1', 'STXBP2', 'SUPV3L1',
    'SUV39H1', 'SUZ12', 'SYK', 'SYNE2', 'SYNGR3', 'SYPL1', 'TARBP1', 'TATDN2',
    'TBC1D31', 'TBC1D9B', 'TBP', 'TBPL1', 'TBX2', 'TBXA2R', 'TCEA2', 'TCEAL4',
    'TCERG1', 'TCFL5', 'TCTA', 'TCTN1', 'TERF2IP', 'TERT', 'TES', 'TESK1',
    'TEX10', 'TFAP2A', 'TFDP1', 'TGFB3', 'TGFBR2', 'THAP11', 'TIAM1', 'TICAM1',
    'TIMELESS', 'TIMM17B', 'TIMM22', 'TIMM9', 'TIMP2', 'TIPARP', 'TJP1', 'TLE1',
    'TLK2', 'TLR4', 'TM9SF2', 'TM9SF3', 'TMCO1', 'TMED10', 'TMEM109', 'TMEM110',
    'TMEM2', 'TMEM5', 'TMEM50A', 'TMEM97', 'TNFRSF21', 'TNIP1', 'TOMM34', 'TOMM70',
    'TOP2A', 'TOPBP1', 'TOR1A', 'TP53', 'TP53BP1', 'TP53BP2', 'TPD52L2', 'TPM1',
    'TRAK2', 'TRAM2', 'TRAP1', 'TRAPPC3', 'TRAPPC6A', 'TRIB1', 'TRIB3', 'TRIM13',
    'TRIM2', 'TSC22D3', 'TSEN2', 'TSKU', 'TSPAN3', 'TSPAN4', 'TSPAN6', 'TSTA3',
    'TUBB6', 'TWF2', 'TXLNA', 'TXNDC9', 'TXNL4B', 'TXNRD1', 'UBE2A', 'UBE2C',
    'UBE2J1', 'UBE2L6', 'UBE3B', 'UBE3C', 'UBQLN2', 'UBR7', 'UFM1', 'UGDH',
    'USP1', 'USP14', 'USP22', 'USP6NL', 'USP7', 'UTP14A', 'VAPB', 'VAT1',
    'VAV3', 'VDAC1', 'VGLL4', 'VPS28', 'VPS72', 'WASF3', 'WASHC4', 'WASHC5',
    'WDR61', 'WDR7', 'WDTC1', 'WFS1', 'WIPF2', 'WRB', 'XBP1', 'XPNPEP1',
    'XPO7', 'YKT6', 'YME1L1', 'YTHDF1', 'ZDHHC6', 'ZFP36', 'ZMIZ1', 'ZMYM2',
    'ZNF131', 'ZNF274', 'ZNF318', 'ZNF395', 'ZNF451', 'ZNF586', 'ZNF589', 'ZW10',
]

MODEL_PREPROCESS = {
    "ridge": "tabular gene expression + Mordred drug descriptors",
    "random_forest": "official-style tabular gene expression + Mordred drug descriptors",
    "lightgbm": "tabular gene expression + Mordred drug descriptors",
    "graphdrp": "official-style drug molecular graph from SMILES + gene expression",
    "simple_linear_nn": "official-style PyTorch MLP on gene expression + Mordred drug descriptors",
}
GRAPHDRP_ALLOWABLE_ATOMS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
]
GRAPHDRP_ALLOWABLE_DEGREES = list(range(11))
GRAPHDRP_ALLOWABLE_TOTAL_HS = list(range(11))
GRAPHDRP_ALLOWABLE_IMPLICIT_VALENCES = list(range(11))


@dataclass
class BenchmarkConfig:
    root: Path
    datasets: List[str]
    folds: List[int]
    models: List[str]
    use_lincs_symbol_genes: bool = True
    top_ge_features: int = 512
    top_mordred_features: int = 512
    max_train_rows: Optional[int] = None
    max_eval_rows: Optional[int] = None
    random_forest_epochs: int = 100
    random_forest_patience: int = 50
    graphdrp_epochs: int = 150
    graphdrp_batch_size: int = 256
    graphdrp_patience: int = 20
    graphdrp_learning_rate: float = 1e-4
    simple_nn_epochs: int = 300
    simple_nn_batch_size: int = 64
    simple_nn_val_batch_size: int = 64
    simple_nn_patience: int = 50
    simple_nn_learning_rate: float = 0.01
    simple_nn_dropout: float = 0.01
    simple_nn_model: str = "default"
    random_state: int = RANDOM_STATE

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / "csa_data" / "raw_data"

    @property
    def x_dir(self) -> Path:
        return self.data_dir / "x_data"

    @property
    def y_dir(self) -> Path:
        return self.data_dir / "y_data"

    @property
    def split_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def out_dir(self) -> Path:
        return self.root / "new_notebook" / "results"


def make_default_config(root: Optional[Path] = None) -> BenchmarkConfig:
    if root is None:
        root = Path.cwd()
        if root.name == "new_notebook":
            root = root.parent
    return BenchmarkConfig(
        root=root,
        datasets=["CCLE"],
        folds=[0],
        models=DEFAULT_MODELS.copy(),
    )


def make_paper_config(root: Optional[Path] = None) -> BenchmarkConfig:
    """Full within-dataset setting: same datasets, splits, and metrics as the paper table."""
    cfg = make_default_config(root)
    cfg.datasets = BENCHMARK_DATASETS.copy()
    cfg.folds = BENCHMARK_FOLDS.copy()
    cfg.models = DEFAULT_MODELS.copy()
    return cfg


def write_benchmark_contract(cfg: BenchmarkConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "fairness_basis": "same dataset, same official split files, same target, same evaluation metrics",
        "analysis": "within_dataset",
        "target": TARGET_COL,
        "datasets": cfg.datasets,
        "folds": cfg.folds,
        "split_files": str(cfg.split_dir / "{dataset}_split_{fold}_{train|val|test}.txt"),
        "primary_metric": "test R2; report mean/std across folds",
        "secondary_metrics": ["RMSE", "MAE", "Pearson"],
        "model_specific_preprocess": {model: MODEL_PREPROCESS.get(model, "custom") for model in cfg.models},
    }
    path = cfg.out_dir / "benchmark_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")


def write_cross_dataset_contract(cfg: BenchmarkConfig, source_datasets: List[str], target_datasets: List[str]) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "fairness_basis": "same benchmark response table, official source split files, target-all evaluation for cross-dataset entries, same target, same metrics",
        "analysis": "cross_dataset",
        "target": TARGET_COL,
        "source_datasets": source_datasets,
        "target_datasets": target_datasets,
        "folds": cfg.folds,
        "train_split": "{source}_split_{fold}_train.txt",
        "val_split": "{source}_split_{fold}_val.txt",
        "test_split_same_dataset": "{source}_split_{fold}_test.txt",
        "test_split_cross_dataset": "{target}_all.txt",
        "primary_metric": "test R2; G[source,target] is mean across folds",
        "secondary_metrics": ["RMSE", "MAE", "Pearson"],
        "aggregates": ["G", "Ga", "Gn", "Gna"],
        "model_specific_preprocess": {model: MODEL_PREPROCESS.get(model, "custom") for model in cfg.models},
    }
    path = cfg.out_dir / "cross_dataset_benchmark_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")


def read_response(cfg: BenchmarkConfig) -> pd.DataFrame:
    path = cfg.y_dir / "response.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    response = pd.read_csv(path, sep="\t", low_memory=False)
    response = response.reset_index().rename(columns={"index": "global_index"})
    response[CELL_ID_COL] = response[CELL_ID_COL].astype(str)
    response[DRUG_ID_COL] = response[DRUG_ID_COL].astype(str)
    return response


def read_gene_expression(cfg: BenchmarkConfig) -> pd.DataFrame:
    path = cfg.x_dir / "cancer_gene_expression.tsv"
    ge = pd.read_csv(path, sep="\t", skiprows=[1, 2], index_col=0, low_memory=False)
    ge.index = ge.index.astype(str)
    ge = ge[~ge.index.duplicated(keep="first")]
    return ge.apply(pd.to_numeric, errors="coerce")


def read_gene_symbols(cfg: BenchmarkConfig) -> Dict[str, str]:
    path = cfg.x_dir / "cancer_gene_expression.tsv"
    meta = pd.read_csv(path, sep="\t", nrows=2, index_col=0, low_memory=False, dtype=str)
    if len(meta) < 2:
        return {}
    return meta.iloc[1].dropna().astype(str).to_dict()


def get_lincs_symbol_list(required: bool = False) -> Optional[List[str]]:
    if LINCS_SYMBOL:
        return list(LINCS_SYMBOL)
    if required:
        raise RuntimeError("Built-in LINCS_SYMBOL list is empty.")
    return None


def select_gene_columns(cfg: BenchmarkConfig, gene_expression: pd.DataFrame, train_cell_ids: Iterable[str]) -> List[str]:
    if cfg.use_lincs_symbol_genes:
        lincs_symbols = get_lincs_symbol_list(required=True)
        ens_to_symbol = read_gene_symbols(cfg)
        symbol_to_ens = {}
        for ens_id, symbol in ens_to_symbol.items():
            if ens_id in gene_expression.columns and symbol not in symbol_to_ens:
                symbol_to_ens[symbol] = ens_id
        lincs_cols = [symbol_to_ens[symbol] for symbol in lincs_symbols if symbol in symbol_to_ens]
        if not lincs_cols:
            raise ValueError("Built-in LINCS_SYMBOL list loaded, but none mapped to gene expression columns.")
        return lincs_cols

    return top_variance_columns(gene_expression, train_cell_ids, cfg.top_ge_features)


def read_mordred(cfg: BenchmarkConfig) -> pd.DataFrame:
    path = cfg.x_dir / "drug_mordred.tsv"
    md = pd.read_csv(path, sep="\t", low_memory=False)
    md[DRUG_ID_COL] = md[DRUG_ID_COL].astype(str)
    md = md.drop_duplicates(DRUG_ID_COL).set_index(DRUG_ID_COL)
    return md.apply(pd.to_numeric, errors="coerce")


def _read_feature_table_with_two_metadata_rows(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", skiprows=[1, 2], index_col=0, low_memory=False)
    table.index = table.index.astype(str)
    table = table[~table.index.duplicated(keep="first")]
    return table.apply(pd.to_numeric, errors="coerce")


def read_mutation_binary(cfg: BenchmarkConfig) -> pd.DataFrame:
    path = cfg.x_dir / "cancer_mutation_count.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    mutation_count = _read_feature_table_with_two_metadata_rows(path)
    return mutation_count.fillna(0).gt(0).astype(np.float32)


def read_cnv_binary(cfg: BenchmarkConfig) -> pd.DataFrame:
    path = cfg.x_dir / "cancer_discretized_copy_number.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    cnv = _read_feature_table_with_two_metadata_rows(path)
    return cnv.fillna(0).ne(0).astype(np.float32)


def read_smiles(cfg: BenchmarkConfig) -> pd.DataFrame:
    path = cfg.x_dir / "drug_SMILES.tsv"
    smiles = pd.read_csv(path, sep="\t", low_memory=False)
    smiles[DRUG_ID_COL] = smiles[DRUG_ID_COL].astype(str)
    return smiles.drop_duplicates(DRUG_ID_COL)


def load_split_indices(cfg: BenchmarkConfig, dataset: str, fold: int, stage: str) -> np.ndarray:
    path = cfg.split_dir / f"{dataset}_split_{fold}_{stage}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.atleast_1d(np.loadtxt(path, dtype=int))


def response_for_split(
    cfg: BenchmarkConfig,
    response: pd.DataFrame,
    dataset: str,
    fold: int,
    stage: str,
) -> pd.DataFrame:
    idx = load_split_indices(cfg, dataset, fold, stage)
    df = response.iloc[idx].copy()
    df = df[df["source"].eq(dataset)]
    df = df.dropna(subset=[TARGET_COL, CELL_ID_COL, DRUG_ID_COL])
    df[CELL_ID_COL] = df[CELL_ID_COL].astype(str)
    df[DRUG_ID_COL] = df[DRUG_ID_COL].astype(str)
    return df


def response_for_dataset_all(cfg: BenchmarkConfig, response: pd.DataFrame, dataset: str) -> pd.DataFrame:
    path = cfg.split_dir / f"{dataset}_all.txt"
    if path.exists():
        idx = np.atleast_1d(np.loadtxt(path, dtype=int))
        df = response.iloc[idx].copy()
    else:
        df = response[response["source"].eq(dataset)].copy()
    df = df[df["source"].eq(dataset)]
    df = df.dropna(subset=[TARGET_COL, CELL_ID_COL, DRUG_ID_COL])
    df[CELL_ID_COL] = df[CELL_ID_COL].astype(str)
    df[DRUG_ID_COL] = df[DRUG_ID_COL].astype(str)
    return df


def top_variance_columns(table: pd.DataFrame, ids: Iterable[str], top_k: Optional[int]) -> List[str]:
    matched = table.loc[table.index.intersection(pd.Index(ids))]
    if matched.empty:
        raise ValueError("No matching feature rows for train IDs")
    variances = matched.var(axis=0, skipna=True).fillna(0)
    variances = variances[variances > 0]
    if top_k is None or top_k >= len(variances):
        return variances.sort_values(ascending=False).index.tolist()
    return variances.nlargest(top_k).index.tolist()


def build_tabular_matrix(
    split_df: pd.DataFrame,
    gene_expression: pd.DataFrame,
    mordred: pd.DataFrame,
    ge_cols: List[str],
    md_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    ids = split_df[[CELL_ID_COL, DRUG_ID_COL, TARGET_COL]].copy()
    ids["row_id"] = np.arange(len(ids))

    ge_part = gene_expression[ge_cols].copy()
    ge_part.columns = [f"ge.{c}" for c in ge_part.columns]
    ge_part = ge_part.reset_index().rename(columns={ge_part.index.name or "index": CELL_ID_COL})

    md_part = mordred[md_cols].copy()
    md_part.columns = [f"mordred.{c}" for c in md_part.columns]
    md_part = md_part.reset_index().rename(columns={md_part.index.name or "index": DRUG_ID_COL})

    data = ids.merge(ge_part, on=CELL_ID_COL, how="inner")
    data = data.merge(md_part, on=DRUG_ID_COL, how="inner")
    data = data.sort_values("row_id").reset_index(drop=True)

    feature_cols = [c for c in data.columns if c.startswith("ge.") or c.startswith("mordred.")]
    X = data[feature_cols]
    y = data[TARGET_COL].astype(float).to_numpy()
    meta = data[[CELL_ID_COL, DRUG_ID_COL, TARGET_COL]].copy()
    return X, y, meta


def maybe_sample(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    max_rows: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if max_rows is None or len(X) <= max_rows:
        return X, y, meta
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=max_rows, replace=False))
    return X.iloc[idx].reset_index(drop=True), y[idx], meta.iloc[idx].reset_index(drop=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mse = mean_squared_error(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan,
    }


def get_tabular_models(cfg: BenchmarkConfig) -> Dict[str, object]:
    models: Dict[str, object] = {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
    }
    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                objective="regression",
                n_estimators=800,
                learning_rate=0.05,
                num_leaves=31,
                n_jobs=-1,
                random_state=cfg.random_state,
                verbose=-1,
            ),
        )
    except ImportError:
        pass
    return models


def fit_transform_tabular_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(X_train)).astype(np.float32)
    x_val = scaler.transform(imputer.transform(X_val)).astype(np.float32)
    x_test = scaler.transform(imputer.transform(X_test)).astype(np.float32)
    return x_train, x_val, x_test


def run_random_forest_official_style_model(
    cfg: BenchmarkConfig,
    dataset: str,
    fold: int,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> List[Dict[str, object]]:
    x_train, x_val, x_test = fit_transform_tabular_features(X_train, X_val, X_test)

    start = time.time()
    model = RandomForestRegressor(
        max_depth=None,
        n_estimators=1,
        warm_start=True,
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    best_loss = math.inf
    best_model = None
    early_stop = 0
    rounds = 0

    while rounds < cfg.random_forest_epochs and early_stop < cfg.random_forest_patience:
        rounds += 1
        model.set_params(n_estimators=rounds)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_loss = mean_squared_error(y_val, val_pred)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model)
            early_stop = 0
        else:
            early_stop += 1

    train_seconds = time.time() - start
    if best_model is None:
        best_model = model

    rows = []
    for stage, X_stage, y_stage in [("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test)]:
        pred = best_model.predict(X_stage)
        rows.append(
            {
                "analysis": "within_dataset",
                "dataset": dataset,
                "fold": fold,
                "stage": stage,
                "model": "random_forest",
                "train_seconds": train_seconds,
                "n_train": len(x_train),
                "n_features": x_train.shape[1],
                "status": "ok",
                **regression_metrics(y_stage, pred),
            }
        )
    return rows


def run_tabular_model(
    model_name: str,
    estimator: object,
    dataset: str,
    fold: int,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> List[Dict[str, object]]:
    start = time.time()
    model = clone(estimator)
    model.fit(X_train, y_train)
    train_seconds = time.time() - start

    rows = []
    for stage, X_stage, y_stage in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = model.predict(X_stage)
        rows.append(
            {
                "analysis": "within_dataset",
                "dataset": dataset,
                "fold": fold,
                "stage": stage,
                "model": model_name,
                "train_seconds": train_seconds,
                "n_train": len(X_train),
                "n_features": X_train.shape[1],
                "status": "ok",
                **regression_metrics(y_stage, pred),
            }
        )
    return rows


def _require_torch_deps(model_name: str):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(f"{model_name} requires torch.") from exc
    return torch, nn, DataLoader, TensorDataset


def run_simple_linear_nn_model(
    cfg: BenchmarkConfig,
    dataset: str,
    fold: int,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> List[Dict[str, object]]:
    torch, nn, DataLoader, TensorDataset = _require_torch_deps("SimpleLinearNN")
    x_train, x_val, x_test = fit_transform_tabular_features(X_train, X_val, X_test)

    torch.manual_seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.simple_nn_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.simple_nn_val_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.simple_nn_val_batch_size, shuffle=False)

    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim: int, dropout_prob: float):
            super().__init__()
            h1 = max(1, int(np.floor(input_dim / 2)))
            h2 = max(1, int(np.floor(input_dim / 4)))
            self.net = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(h1, h2),
                nn.LeakyReLU(),
                nn.Linear(h2, 1),
            )

        def forward(self, x):
            return self.net(x)

    class LinearRegressionModelSmall(nn.Module):
        def __init__(self, input_dim: int, dropout_prob: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(128, 8),
                nn.LeakyReLU(),
                nn.Linear(8, 1),
            )

        def forward(self, x):
            return self.net(x)

    class LinearRegressionModelTiny(nn.Module):
        def __init__(self, input_dim: int, dropout_prob: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Dropout(p=dropout_prob),
                nn.Linear(input_dim, 8),
                nn.LeakyReLU(),
                nn.Linear(8, 1),
            )

        def forward(self, x):
            return self.net(x)

    class LinearRegressionModelLarge(nn.Module):
        def __init__(self, input_dim: int, dropout_prob: float):
            super().__init__()
            h1 = max(1, int(np.floor(input_dim / 1.5)))
            h2 = max(1, int(np.floor(input_dim / 2)))
            h3 = max(1, int(np.floor(input_dim / 3)))
            h4 = max(1, int(np.floor(input_dim / 4)))
            h5 = max(1, int(np.floor(input_dim / 8)))
            self.net = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(h1, h2),
                nn.LeakyReLU(),
                nn.Linear(h2, h3),
                nn.LeakyReLU(),
                nn.Linear(h3, h4),
                nn.LeakyReLU(),
                nn.Linear(h4, h5),
                nn.LeakyReLU(),
                nn.Linear(h5, 1),
            )

        def forward(self, x):
            return self.net(x)

    model_classes = {
        "default": LinearRegressionModel,
        "small": LinearRegressionModelSmall,
        "tiny": LinearRegressionModelTiny,
        "large": LinearRegressionModelLarge,
    }
    simple_nn_arch = cfg.simple_nn_model
    if simple_nn_arch in {"simple_linear_nn", "simplelinearnn", "SimpleLinearNN"}:
        simple_nn_arch = "default"

    model_class = model_classes.get(simple_nn_arch)
    if model_class is None:
        raise ValueError(
            f"Unknown simple_nn_model: {cfg.simple_nn_model}. "
            "Use one of: default, small, tiny, large."
        )

    model = model_class(x_train.shape[1], cfg.simple_nn_dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.simple_nn_learning_rate)

    def predict_loader(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                pred = model(batch_x.to(device)).view(-1)
                preds.append(pred.cpu().numpy())
                trues.append(batch_y.numpy())
        model.train()
        return np.concatenate(trues), np.concatenate(preds)

    best_loss = math.inf
    best_state = None
    bad_epochs = 0
    start = time.time()
    for _epoch in range(cfg.simple_nn_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).view(-1, 1)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        yv, pv = predict_loader(val_loader)
        val_loss = mean_squared_error(yv, pv)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.simple_nn_patience:
                break

    train_seconds = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    rows = []
    for stage, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        y_true, y_pred = predict_loader(loader)
        rows.append(
            {
                "analysis": "within_dataset",
                "dataset": dataset,
                "fold": fold,
                "stage": stage,
                "model": "simple_linear_nn",
                "train_seconds": train_seconds,
                "n_train": len(x_train),
                "n_features": x_train.shape[1],
                "status": "ok",
                **regression_metrics(y_true, y_pred),
            }
        )
    return rows


def _require_graphdrp_deps():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from rdkit import Chem
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINConv, global_add_pool
    except ImportError as exc:
        raise ImportError(
            "GraphDRP requires torch, torch-geometric, and rdkit. "
            "Install them before running the graphdrp model."
        ) from exc
    return torch, nn, F, Chem, Data, DataLoader, GINConv, global_add_pool


def run_graphdrp_model(
    cfg: BenchmarkConfig,
    dataset: str,
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gene_expression: pd.DataFrame,
    ge_cols: List[str],
    smiles: pd.DataFrame,
) -> List[Dict[str, object]]:
    torch, nn, F, Chem, Data, DataLoader, GINConv, global_add_pool = _require_graphdrp_deps()

    import random

    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.random_state)

    train_cells = set(train_df[CELL_ID_COL].astype(str))
    all_cells = set(train_df[CELL_ID_COL].astype(str)) | set(val_df[CELL_ID_COL].astype(str)) | set(
        test_df[CELL_ID_COL].astype(str)
    )
    all_drugs = set(train_df[DRUG_ID_COL].astype(str)) | set(val_df[DRUG_ID_COL].astype(str)) | set(
        test_df[DRUG_ID_COL].astype(str)
    )

    ge = gene_expression.loc[gene_expression.index.intersection(pd.Index(all_cells)), ge_cols].copy()
    train_ge = ge.loc[ge.index.intersection(pd.Index(train_cells))]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train_ge)
    scaler.fit(train_imputed)
    ge_scaled = scaler.transform(imputer.transform(ge)).astype(np.float32)
    cell_feat = dict(zip(ge.index.astype(str), ge_scaled))

    smiles_col = "canSMILES" if "canSMILES" in smiles.columns else "smiles"
    smiles_dict = dict(zip(smiles[DRUG_ID_COL].astype(str), smiles[smiles_col].astype(str)))

    def one_hot_unknown(value, choices):
        if value not in choices:
            value = choices[-1]
        return [float(value == choice) for choice in choices]

    def one_hot_strict(value, choices):
        if value not in choices:
            raise ValueError(f"input {value} not in allowable set {choices}")
        return [float(value == choice) for choice in choices]

    def atom_features(atom):
        features = np.array(
            one_hot_unknown(atom.GetSymbol(), GRAPHDRP_ALLOWABLE_ATOMS)
            + one_hot_strict(atom.GetDegree(), GRAPHDRP_ALLOWABLE_DEGREES)
            + one_hot_unknown(atom.GetTotalNumHs(), GRAPHDRP_ALLOWABLE_TOTAL_HS)
            + one_hot_unknown(atom.GetImplicitValence(), GRAPHDRP_ALLOWABLE_IMPLICIT_VALENCES)
            + [float(atom.GetIsAromatic())]
        ).astype(np.float32)
        return features / features.sum()

    def mol_to_graph(drug_id: str):
        smi = smiles_dict.get(drug_id)
        if not smi or smi == "nan":
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        try:
            atom_feature_matrix = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
        except ValueError:
            return None
        x = torch.tensor(atom_feature_matrix, dtype=torch.float)
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append((i, j))
            edges.append((j, i))
        if not edges:
            return None
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return {"x": x, "edge_index": edge_index}

    graph_cache = {drug_id: mol_to_graph(drug_id) for drug_id in all_drugs}

    y_mean = float(train_df[TARGET_COL].mean())
    y_std = float(train_df[TARGET_COL].std())
    if not np.isfinite(y_std) or y_std == 0:
        y_std = 1.0

    def build_dataset(df: pd.DataFrame):
        if cfg.max_train_rows is not None and len(df) > cfg.max_train_rows and df is train_df:
            idx = np.sort(rng.choice(len(df), size=cfg.max_train_rows, replace=False))
            df = df.iloc[idx]
        rows = []
        for _, row in df.iterrows():
            drug_id = str(row[DRUG_ID_COL])
            cell_id = str(row[CELL_ID_COL])
            graph = graph_cache.get(drug_id)
            cell = cell_feat.get(cell_id)
            y = row[TARGET_COL]
            if graph is None or cell is None or not np.isfinite(y):
                continue
            rows.append(
                Data(
                    x=graph["x"].clone(),
                    edge_index=graph["edge_index"].clone(),
                    target=torch.tensor(cell, dtype=torch.float).unsqueeze(0),
                    y=torch.tensor([float(y)], dtype=torch.float),
                )
            )
        return rows

    data_train = build_dataset(train_df)
    data_val = build_dataset(val_df)
    data_test = build_dataset(test_df)
    if not data_train or not data_val or not data_test:
        raise ValueError("GraphDRP has an empty train/val/test dataset after feature filtering.")

    class GraphDRPNet(nn.Module):
        def __init__(self, node_dim: int, cell_dim: int):
            super().__init__()
            dim = 32
            output_dim = 128
            self.relu = nn.ReLU()
            self.dropout_graph = nn.Dropout(0.2)
            self.dropout_fusion = nn.Dropout(0.5)

            self.conv1 = GINConv(nn.Sequential(nn.Linear(node_dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
            self.bn1 = nn.BatchNorm1d(dim)
            self.conv2 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
            self.bn2 = nn.BatchNorm1d(dim)
            self.conv3 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
            self.bn3 = nn.BatchNorm1d(dim)
            self.conv4 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
            self.bn4 = nn.BatchNorm1d(dim)
            self.conv5 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
            self.bn5 = nn.BatchNorm1d(dim)
            self.fc1_xd = nn.Linear(dim, output_dim)

            self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8)
            self.pool_xt_1 = nn.MaxPool1d(3)
            self.conv_xt_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
            self.pool_xt_2 = nn.MaxPool1d(3)
            self.conv_xt_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
            self.pool_xt_3 = nn.MaxPool1d(3)
            with torch.no_grad():
                dummy_target = torch.zeros(1, 1, cell_dim)
                conv_xt = self.pool_xt_1(F.relu(self.conv_xt_1(dummy_target)))
                conv_xt = self.pool_xt_2(F.relu(self.conv_xt_2(conv_xt)))
                conv_xt = self.pool_xt_3(F.relu(self.conv_xt_3(conv_xt)))
                self.in_dim = conv_xt.shape[1] * conv_xt.shape[2]
            self.fc1_xt = nn.Linear(self.in_dim, output_dim)

            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 128)
            self.out = nn.Linear(128, 1)

        def forward(self, batch):
            x = F.relu(self.conv1(batch.x, batch.edge_index))
            x = self.bn1(x)
            x = F.relu(self.conv2(x, batch.edge_index))
            x = self.bn2(x)
            x = F.relu(self.conv3(x, batch.edge_index))
            x = self.bn3(x)
            x = F.relu(self.conv4(x, batch.edge_index))
            x = self.bn4(x)
            x = F.relu(self.conv5(x, batch.edge_index))
            x = self.bn5(x)
            drug = global_add_pool(x, batch.batch)
            drug = F.relu(self.fc1_xd(drug))
            drug = self.dropout_graph(drug)

            target = batch.target[:, None, :]
            conv_xt = self.pool_xt_1(F.relu(self.conv_xt_1(target)))
            conv_xt = self.pool_xt_2(F.relu(self.conv_xt_2(conv_xt)))
            conv_xt = self.pool_xt_3(F.relu(self.conv_xt_3(conv_xt)))
            cell = self.fc1_xt(conv_xt.reshape(-1, self.in_dim))

            fusion = torch.cat((drug, cell), 1)
            fusion = self.dropout_fusion(self.relu(self.fc1(fusion)))
            fusion = self.dropout_fusion(self.relu(self.fc2(fusion)))
            return torch.sigmoid(self.out(fusion)).squeeze(-1)

    train_loader = DataLoader(data_train, batch_size=cfg.graphdrp_batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=cfg.graphdrp_batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=cfg.graphdrp_batch_size, shuffle=False)

    model = GraphDRPNet(data_train[0].x.shape[1], data_train[0].target.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.graphdrp_learning_rate)
    loss_fn = nn.MSELoss()

    def eval_loader(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch)
                preds.append(pred.cpu().numpy())
                trues.append(batch.y.view(-1).cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        return trues, preds

    best_val = math.inf
    best_state = None
    bad_epochs = 0
    start = time.time()
    for _epoch in range(cfg.graphdrp_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()
        yv, pv = eval_loader(val_loader)
        val_mse = float(mean_squared_error(yv, pv))
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= cfg.graphdrp_patience:
            break
    train_seconds = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    rows = []
    for stage, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        y_true, y_pred = eval_loader(loader)
        rows.append(
            {
                "analysis": "within_dataset",
                "dataset": dataset,
                "fold": fold,
                "stage": stage,
                "model": "graphdrp",
                "train_seconds": train_seconds,
                "n_train": len(data_train),
                "n_features": data_train[0].target.shape[1],
                "status": "ok",
                **regression_metrics(y_true, y_pred),
            }
        )
    return rows


def run_within_dataset_benchmark(cfg: BenchmarkConfig) -> pd.DataFrame:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_contract(cfg)

    response = read_response(cfg)
    gene_expression = read_gene_expression(cfg)
    mordred = read_mordred(cfg)
    smiles = read_smiles(cfg)

    tabular_models = get_tabular_models(cfg)
    results: List[Dict[str, object]] = []

    for dataset in cfg.datasets:
        for fold in cfg.folds:
            print("=" * 80)
            print(f"Dataset={dataset} | fold={fold}")
            train_df = response_for_split(cfg, response, dataset, fold, "train")
            val_df = response_for_split(cfg, response, dataset, fold, "val")
            test_df = response_for_split(cfg, response, dataset, fold, "test")
            split_sizes = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}

            ge_cols = select_gene_columns(cfg, gene_expression, train_df[CELL_ID_COL].unique())
            md_cols = top_variance_columns(mordred, train_df[DRUG_ID_COL].unique(), cfg.top_mordred_features)

            X_train, y_train, meta_train = build_tabular_matrix(train_df, gene_expression, mordred, ge_cols, md_cols)
            X_val, y_val, meta_val = build_tabular_matrix(val_df, gene_expression, mordred, ge_cols, md_cols)
            X_test, y_test, meta_test = build_tabular_matrix(test_df, gene_expression, mordred, ge_cols, md_cols)

            X_train, y_train, meta_train = maybe_sample(
                X_train, y_train, meta_train, cfg.max_train_rows, cfg.random_state
            )
            X_val, y_val, meta_val = maybe_sample(X_val, y_val, meta_val, cfg.max_eval_rows, cfg.random_state)
            X_test, y_test, meta_test = maybe_sample(X_test, y_test, meta_test, cfg.max_eval_rows, cfg.random_state)

            print(
                f"Rows usable: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,} | "
                f"tabular features={X_train.shape[1]:,}"
            )

            for model_name in cfg.models:
                if model_name in {"ridge", "lightgbm"}:
                    if model_name not in tabular_models:
                        results.append(
                            {
                                "analysis": "within_dataset",
                                "dataset": dataset,
                                "fold": fold,
                                "stage": "test",
                                "model": model_name,
                                "preprocess": MODEL_PREPROCESS.get(model_name, "custom"),
                                "status": "skipped_missing_dependency",
                            }
                        )
                        print(f"  Skipped {model_name}: missing dependency")
                        continue
                    print(f"  Training {model_name}")
                    rows = run_tabular_model(
                        model_name,
                        tabular_models[model_name],
                        dataset,
                        fold,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                    )
                elif model_name == "random_forest":
                    print("  Training random_forest")
                    rows = run_random_forest_official_style_model(
                        cfg,
                        dataset,
                        fold,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        X_test,
                        y_test,
                    )
                elif model_name == "simple_linear_nn":
                    print("  Training simple_linear_nn")
                    try:
                        rows = run_simple_linear_nn_model(
                            cfg,
                            dataset,
                            fold,
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                        )
                    except ImportError as exc:
                        rows = [
                            {
                                "analysis": "within_dataset",
                                "dataset": dataset,
                                "fold": fold,
                                "stage": "test",
                                "model": "simple_linear_nn",
                                "status": "skipped_missing_dependency",
                                "error": str(exc),
                            }
                        ]
                        print(f"  Skipped simple_linear_nn: {exc}")
                elif model_name == "graphdrp":
                    print("  Training graphdrp")
                    try:
                        rows = run_graphdrp_model(
                            cfg,
                            dataset,
                            fold,
                            train_df,
                            val_df,
                            test_df,
                            gene_expression,
                            ge_cols,
                            smiles,
                        )
                    except ImportError as exc:
                        rows = [
                            {
                                "analysis": "within_dataset",
                                "dataset": dataset,
                                "fold": fold,
                                "stage": "test",
                                "model": "graphdrp",
                                "status": "skipped_missing_dependency",
                                "error": str(exc),
                            }
                        ]
                        print(f"  Skipped graphdrp: {exc}")
                    except Exception as exc:
                        rows = [
                            {
                                "analysis": "within_dataset",
                                "dataset": dataset,
                                "fold": fold,
                                "stage": "test",
                                "model": "graphdrp",
                                "status": "failed",
                                "error": repr(exc),
                            }
                        ]
                        print(f"  Failed graphdrp: {exc!r}")
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                results.extend(rows)
                for row in rows:
                    row.setdefault("preprocess", MODEL_PREPROCESS.get(model_name, "custom"))
                    row.setdefault("split_train_rows", split_sizes["train"])
                    row.setdefault("split_val_rows", split_sizes["val"])
                    row.setdefault("split_test_rows", split_sizes["test"])
                    if row.get("status") == "ok":
                        print(
                            f"    {row['stage']}: n={row['n']:,} RMSE={row['rmse']:.4f} "
                            f"MAE={row['mae']:.4f} R2={row['r2']:.4f} Pearson={row['pearson']:.4f}"
                        )

    results_df = pd.DataFrame(results)
    results_path = cfg.out_dir / "within_dataset_4models_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")
    return results_df


def run_cross_dataset_benchmark(
    cfg: BenchmarkConfig,
    source_datasets: Optional[List[str]] = None,
    target_datasets: Optional[List[str]] = None,
    only_cross_dataset: bool = False,
) -> pd.DataFrame:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    source_datasets = source_datasets or cfg.datasets
    target_datasets = target_datasets or cfg.datasets
    write_cross_dataset_contract(cfg, source_datasets, target_datasets)

    response = read_response(cfg)
    gene_expression = read_gene_expression(cfg)
    mordred = read_mordred(cfg)
    smiles = read_smiles(cfg)

    tabular_models = get_tabular_models(cfg)
    results: List[Dict[str, object]] = []

    for source_dataset in source_datasets:
        for fold in cfg.folds:
            print("=" * 90)
            print(f"Source={source_dataset} | fold={fold}")
            train_df = response_for_split(cfg, response, source_dataset, fold, "train")
            val_df = response_for_split(cfg, response, source_dataset, fold, "val")

            ge_cols = select_gene_columns(cfg, gene_expression, train_df[CELL_ID_COL].unique())
            md_cols = top_variance_columns(mordred, train_df[DRUG_ID_COL].unique(), cfg.top_mordred_features)

            X_train, y_train, meta_train = build_tabular_matrix(train_df, gene_expression, mordred, ge_cols, md_cols)
            X_val, y_val, meta_val = build_tabular_matrix(val_df, gene_expression, mordred, ge_cols, md_cols)

            X_train, y_train, meta_train = maybe_sample(
                X_train, y_train, meta_train, cfg.max_train_rows, cfg.random_state
            )
            X_val, y_val, meta_val = maybe_sample(X_val, y_val, meta_val, cfg.max_eval_rows, cfg.random_state)

            for target_dataset in target_datasets:
                if only_cross_dataset and source_dataset == target_dataset:
                    continue

                if source_dataset == target_dataset:
                    target_df = response_for_split(cfg, response, source_dataset, fold, "test")
                    target_split_name = f"{source_dataset}_split_{fold}_test.txt"
                else:
                    target_df = response_for_dataset_all(cfg, response, target_dataset)
                    target_split_name = f"{target_dataset}_all.txt"

                X_target, y_target, meta_target = build_tabular_matrix(
                    target_df, gene_expression, mordred, ge_cols, md_cols
                )
                X_target, y_target, meta_target = maybe_sample(
                    X_target, y_target, meta_target, cfg.max_eval_rows, cfg.random_state
                )

                split_sizes = {
                    "train": len(train_df),
                    "val": len(val_df),
                    "target": len(target_df),
                }

                print(
                    f"Target={target_dataset} | train={len(X_train):,}, val={len(X_val):,}, "
                    f"target={len(X_target):,} | tabular features={X_train.shape[1]:,}"
                )

                for model_name in cfg.models:
                    if model_name in {"ridge", "lightgbm"}:
                        if model_name not in tabular_models:
                            rows = [
                                {
                                    "analysis": "cross_dataset",
                                    "source_dataset": source_dataset,
                                    "target_dataset": target_dataset,
                                    "fold": fold,
                                    "stage": "test",
                                    "model": model_name,
                                    "preprocess": MODEL_PREPROCESS.get(model_name, "custom"),
                                    "status": "skipped_missing_dependency",
                                }
                            ]
                            print(f"  Skipped {model_name}: missing dependency")
                        else:
                            print(f"  Training {model_name}")
                            rows = run_tabular_model(
                                model_name,
                                tabular_models[model_name],
                                source_dataset,
                                fold,
                                X_train,
                                y_train,
                                X_val,
                                y_val,
                                X_target,
                                y_target,
                            )
                    elif model_name == "random_forest":
                        print("  Training random_forest")
                        rows = run_random_forest_official_style_model(
                            cfg,
                            source_dataset,
                            fold,
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_target,
                            y_target,
                        )
                    elif model_name == "simple_linear_nn":
                        print("  Training simple_linear_nn")
                        try:
                            rows = run_simple_linear_nn_model(
                                cfg,
                                source_dataset,
                                fold,
                                X_train,
                                y_train,
                                X_val,
                                y_val,
                                X_target,
                                y_target,
                            )
                        except ImportError as exc:
                            rows = [
                                {
                                    "analysis": "cross_dataset",
                                    "source_dataset": source_dataset,
                                    "target_dataset": target_dataset,
                                    "fold": fold,
                                    "stage": "test",
                                    "model": "simple_linear_nn",
                                    "status": "skipped_missing_dependency",
                                    "error": str(exc),
                                }
                            ]
                            print(f"  Skipped simple_linear_nn: {exc}")
                    elif model_name == "graphdrp":
                        print("  Training graphdrp")
                        try:
                            rows = run_graphdrp_model(
                                cfg,
                                source_dataset,
                                fold,
                                train_df,
                                val_df,
                                target_df,
                                gene_expression,
                                ge_cols,
                                smiles,
                            )
                        except ImportError as exc:
                            rows = [
                                {
                                    "analysis": "cross_dataset",
                                    "source_dataset": source_dataset,
                                    "target_dataset": target_dataset,
                                    "fold": fold,
                                    "stage": "test",
                                    "model": "graphdrp",
                                    "status": "skipped_missing_dependency",
                                    "error": str(exc),
                                }
                            ]
                            print(f"  Skipped graphdrp: {exc}")
                        except Exception as exc:
                            rows = [
                                {
                                    "analysis": "cross_dataset",
                                    "source_dataset": source_dataset,
                                    "target_dataset": target_dataset,
                                    "fold": fold,
                                    "stage": "test",
                                    "model": "graphdrp",
                                    "status": "failed",
                                    "error": repr(exc),
                                }
                            ]
                            print(f"  Failed graphdrp: {exc!r}")
                    else:
                        raise ValueError(f"Unknown model: {model_name}")

                    for row in rows:
                        row["analysis"] = "cross_dataset"
                        row["source_dataset"] = source_dataset
                        row["target_dataset"] = target_dataset
                        row["target_split_file"] = target_split_name
                        row.setdefault("preprocess", MODEL_PREPROCESS.get(model_name, "custom"))
                        row.setdefault("split_train_rows", split_sizes["train"])
                        row.setdefault("split_val_rows", split_sizes["val"])
                        row.setdefault("split_target_rows", split_sizes["target"])
                        if row.get("stage") == "val":
                            row["target_dataset"] = source_dataset
                            row["target_split_file"] = f"{source_dataset}_split_{fold}_val.txt"
                        if row.get("status") == "ok":
                            print(
                                f"    {row['stage']}: n={row['n']:,} RMSE={row['rmse']:.4f} "
                                f"MAE={row['mae']:.4f} R2={row['r2']:.4f} Pearson={row['pearson']:.4f}"
                            )
                    results.extend(rows)

    results_df = pd.DataFrame(results)
    results_path = cfg.out_dir / "cross_dataset_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")
    return results_df


def summarize_results(results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ok = results[results.get("status", "ok").eq("ok") & results["stage"].eq("test")].copy()
    if ok.empty:
        return pd.DataFrame()
    summary = (
        ok.groupby(["dataset", "model"])
        .agg(
            folds=("fold", "nunique"),
            n_test_mean=("n", "mean"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            train_seconds_mean=("train_seconds", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "r2_mean"], ascending=[True, False])
    )
    path = out_dir / "within_dataset_4models_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")
    return summary


def summarize_cross_dataset_results(results: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ok = results[results.get("status", "ok").eq("ok") & results["stage"].eq("test")].copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        ok.groupby(["model", "source_dataset", "target_dataset"])
        .agg(
            folds=("fold", "nunique"),
            n_test_mean=("n", "mean"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            pearson_mean=("pearson", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
        )
        .reset_index()
        .sort_values(["model", "source_dataset", "target_dataset"])
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "cross_dataset_summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate_rows = []
    for model_name, model_df in summary.groupby("model"):
        r2_matrix = model_df.pivot(index="source_dataset", columns="target_dataset", values="r2_mean")
        r2_std_matrix = model_df.pivot(index="source_dataset", columns="target_dataset", values="r2_std")
        r2_matrix.to_csv(out_dir / f"cross_dataset_G_r2_mean_{model_name}.csv")
        r2_std_matrix.to_csv(out_dir / f"cross_dataset_G_r2_std_{model_name}.csv")

        diag = {dataset: r2_matrix.loc[dataset, dataset] for dataset in r2_matrix.index if dataset in r2_matrix.columns}
        normalized = r2_matrix.copy()
        for source_dataset, source_within_r2 in diag.items():
            if pd.notna(source_within_r2) and source_within_r2 != 0:
                normalized.loc[source_dataset] = normalized.loc[source_dataset] / source_within_r2
        normalized.to_csv(out_dir / f"cross_dataset_Gn_r2_{model_name}.csv")

        for source_dataset in r2_matrix.index:
            cross_targets = [target for target in r2_matrix.columns if target != source_dataset]
            ga = r2_matrix.loc[source_dataset, cross_targets].mean(skipna=True)
            gna = normalized.loc[source_dataset, cross_targets].mean(skipna=True)
            aggregate_rows.append(
                {
                    "model": model_name,
                    "source_dataset": source_dataset,
                    "Ga_mean_cross_r2": ga,
                    "Gna_mean_normalized_cross_r2": gna,
                    "within_r2": diag.get(source_dataset, np.nan),
                    "n_cross_targets": int(r2_matrix.loc[source_dataset, cross_targets].notna().sum()),
                }
            )

    aggregates = pd.DataFrame(aggregate_rows).sort_values(["model", "Ga_mean_cross_r2"], ascending=[True, False])
    aggregates_path = out_dir / "cross_dataset_aggregates_Ga_Gna.csv"
    aggregates.to_csv(aggregates_path, index=False)
    print(f"Saved: {summary_path}")
    print(f"Saved: {aggregates_path}")
    return summary, aggregates


def plot_cross_dataset_g_matrix(
    summary: pd.DataFrame,
    model_name: str,
    dataset_order: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    metric_col: str = "r2_mean",
    std_col: str = "r2_std",
    title: Optional[str] = None,
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: Tuple[float, float] = (7.0, 5.8),
    dpi: int = 200,
):
    """Plot a paper-style cross-dataset G matrix with mean and std in each cell."""
    import os
    import tempfile

    mpl_config_dir = Path(tempfile.gettempdir()) / "matplotlib"
    xdg_cache_dir = Path(tempfile.gettempdir()) / "font-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    if summary.empty:
        raise ValueError("summary is empty. Run summarize_cross_dataset_results first.")
    if metric_col not in summary.columns:
        raise ValueError(f"Missing column: {metric_col}")

    model_df = summary[summary["model"].eq(model_name)].copy()
    if model_df.empty:
        available = sorted(summary["model"].dropna().unique())
        raise ValueError(f"Model {model_name!r} not found. Available models: {available}")

    if dataset_order is None:
        seen = list(dict.fromkeys(model_df["source_dataset"].tolist() + model_df["target_dataset"].tolist()))
        dataset_order = [dataset for dataset in BENCHMARK_DATASETS if dataset in seen]
        dataset_order += [dataset for dataset in seen if dataset not in dataset_order]

    mean_matrix = (
        model_df.pivot(index="source_dataset", columns="target_dataset", values=metric_col)
        .reindex(index=dataset_order, columns=dataset_order)
    )
    std_matrix = None
    if std_col in model_df.columns:
        std_matrix = (
            model_df.pivot(index="source_dataset", columns="target_dataset", values=std_col)
            .reindex(index=dataset_order, columns=dataset_order)
        )

    color_values = mean_matrix.astype(float).clip(lower=vmin, upper=vmax).to_numpy()
    masked_values = np.ma.masked_invalid(color_values)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(masked_values, cmap=cmap, vmin=vmin, vmax=vmax)
    im.cmap.set_bad(color="#f3f3f3")

    ax.set_xticks(np.arange(len(dataset_order)))
    ax.set_xticklabels(dataset_order, rotation=0)
    ax.set_yticks(np.arange(len(dataset_order)))
    ax.set_yticklabels(dataset_order)
    ax.set_xlabel("Target Dataset")
    ax.set_ylabel("Source Dataset")
    ax.set_title(title or f"G matrix for {model_name}")

    for i, source_dataset in enumerate(dataset_order):
        for j, target_dataset in enumerate(dataset_order):
            value = mean_matrix.loc[source_dataset, target_dataset]
            if pd.isna(value):
                text = ""
            else:
                std_value = np.nan if std_matrix is None else std_matrix.loc[source_dataset, target_dataset]
                text = f"{value:.3g}"
                if pd.notna(std_value):
                    text += f"\n({std_value:.3g})"
            text_color = "white" if pd.notna(value) and value >= (vmin + vmax) / 2 else "#333333"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(dataset_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dataset_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$R^2$ score")
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_ticklabels([f"{vmin:g}", f"{(vmin + vmax) / 2:g}", f"\u2264 {vmax:g}"])

    fig.tight_layout()
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"cross_dataset_G_matrix_{model_name}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig, ax


def plot_all_cross_dataset_g_matrices(
    summary: pd.DataFrame,
    dataset_order: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    **plot_kwargs,
) -> Dict[str, Tuple[object, object]]:
    figures = {}
    for model_name in sorted(summary["model"].dropna().unique()):
        figures[model_name] = plot_cross_dataset_g_matrix(
            summary,
            model_name,
            dataset_order=dataset_order,
            out_dir=out_dir,
            **plot_kwargs,
        )
    return figures


if __name__ == "__main__":
    config = make_default_config()
    results_df = run_within_dataset_benchmark(config)
    print(summarize_results(results_df, config.out_dir))
