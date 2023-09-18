import pandas as pd

# List of input files and output files
input_files = [
    "CD16p_Mono_nominal.txt",
    "CL_Mono_nominal.txt",
    "CM_CD8_nominal.txt",
    "DN_B_nominal.txt",
    "EM_CD8_nominal.txt",
    "Fr_I_nTreg_nominal.txt",
    "Fr_II_eTreg_nominal.txt",
    "Fr_III_T_nominal.txt",
    "Int_Mono_nominal.txt",
    "LDG_nominal.txt",
    "mDC_nominal.txt",
    "Mem_CD4_nominal.txt",
    "Mem_CD8_nominal.txt",
    "Naive_B_nominal.txt",
    "Naive_CD4_nominal.txt",
    "Naive_CD8_nominal.txt",
    "NC_Mono_nominal.txt",
    "Neu_nominal.txt",
    "NK_nominal.txt",
    "pDC_nominal.txt",
    "Plasmablast_nominal.txt",
    "SM_B_nominal.txt",
    "TEMRA_CD8_nominal.txt",
    "Tfh_nominal.txt",
    "Th17_nominal.txt",
    "Th1_nominal.txt",
    "Th2_nominal.txt",
    "USM_B_nominal.txt"
]

for input_file in input_files:
    # Read the input file into a DataFrame
    df = pd.read_csv(input_file, sep='\t')

    # Filter rows where the 12th column (index 11) is greater than 0.1
    filtered_df = df[df.iloc[:, 11] > 0.1]

    # Select columns 6 to 10 and write the result to an output file
    filtered_df.iloc[:, 5:10].to_csv(f"{input_file.replace('.txt', '_P_value.txt')}", sep='\t', index=False)
