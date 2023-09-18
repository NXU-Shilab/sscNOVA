import pandas as pd
import os

# List of input files and output files
input_files = [
    "CD16p_Mono_nominal_P_value_2.txt",
    "CL_Mono_nominal_P_value_2.txt",
    "CM_CD8_nominal_P_value_2.txt",
    "DN_B_nominal_P_value_2.txt",
    "EM_CD8_nominal_P_value_2.txt",
    "Fr_II_eTreg_nominal_P_value_2.txt",
    "Fr_III_T_nominal_P_value_2.txt",
    "Fr_I_nTreg_nominal_P_value_2.txt",
    "Int_Mono_nominal_P_value_2.txt",
    "LDG_nominal_P_value_2.txt",
    "mDC_nominal_P_value_2.txt",
    "Mem_CD4_nominal_P_value_2.txt",
    "Mem_CD8_nominal_P_value_2.txt",
    "Naive_B_nominal_P_value_2.txt",
    "Naive_CD4_nominal_P_value_2.txt",
    "Naive_CD8_nominal_P_value_2.txt",
    "NC_Mono_nominal_P_value_2.txt",
    "Neu_nominal_P_value_2.txt",
    "NK_nominal_P_value_2.txt",
    "pDC_nominal_P_value_2.txt",
    "Plasmablast_nominal_P_value_2.txt",
    "SM_B_nominal_P_value_2.txt",
    "TEMRA_CD8_nominal_P_value_2.txt",
    "Tfh_nominal_P_value_2.txt",
    "Th17_nominal_P_value_2.txt",
    "Th1_nominal_P_value_2.txt",
    "Th2_nominal_P_value_2.txt",
    "USM_B_nominal_P_value_2.txt"
]

for input_file in input_files:
    # Define output file name
    output_file = os.path.splitext(input_file)[0] + "_2.vcf"

    # Read the input file into a DataFrame
    df = pd.read_csv(input_file, sep='\t', header=None)

    # Select columns 4, 5, 1, 2, and 3, and write the result to the output file
    df.iloc[:, [3, 4, 0, 1, 2]].drop_duplicates().to_csv(output_file, sep='\t', header=False, index=False)
