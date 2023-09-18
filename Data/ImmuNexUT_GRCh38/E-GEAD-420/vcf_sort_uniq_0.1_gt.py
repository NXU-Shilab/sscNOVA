import pandas as pd
import os

# List of input files and output files
input_files = [
    "CD16p_Mono_SNP.txt",
    "CL_Mono_SNP.txt",
    "CM_CD8_SNP.txt",
    "DN_B_SNP.txt",
    "EM_CD8_SNP.txt",
    "Fr_I_nTreg_SNP.txt",
    "Fr_II_eTreg_SNP.txt",
    "Fr_III_T_SNP.txt",
    "Int_Mono_SNP.txt",
    "LDG_SNP.txt",
    "mDC_SNP.txt",
    "Mem_CD4_SNP.txt",
    "Mem_CD8_SNP.txt",
    "Naive_B_SNP.txt",
    "Naive_CD4_SNP.txt",
    "Naive_CD8_SNP.txt",
    "NC_Mono_SNP.txt",
    "Neu_SNP.txt",
    "NK_SNP.txt",
    "pDC_SNP.txt",
    "Plasmablast_SNP.txt",
    "SM_B_SNP.txt",
    "TEMRA_CD8_SNP.txt",
    "Tfh_SNP.txt",
    "Th17_SNP.txt",
    "Th1_SNP.txt",
    "Th2_SNP.txt",
    "USM_B_SNP.txt"
]

for input_file in input_files:
    # Define output file name
    output_file = os.path.splitext(input_file)[0] + ".vcf"

    # Read the input file into a DataFrame, skipping the first line
    df = pd.read_csv(input_file, sep='\t', skiprows=1, header=None)

    # Select columns 4, 5, 1, 2, and 3, and write the result to the output file
    df.iloc[:, [3, 4, 0, 1, 2]].to_csv(output_file, sep='\t', header=False, index=False)
