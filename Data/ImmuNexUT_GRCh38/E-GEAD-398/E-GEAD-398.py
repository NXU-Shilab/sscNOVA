import os

# Define a list of file names
file_names = [
    "CD16p_Mono_conditional_eQTL_FDR0.05.txt",
    "CL_Mono_conditional_eQTL_FDR0.05.txt",
    "CM_CD8_conditional_eQTL_FDR0.05.txt",
    "DN_B_conditional_eQTL_FDR0.05.txt",
    "EM_CD8_conditional_eQTL_FDR0.05.txt",
    "Fr_I_nTreg_conditional_eQTL_FDR0.05.txt",
    "Fr_II_eTreg_conditional_eQTL_FDR0.05.txt",
    "Fr_III_T_conditional_eQTL_FDR0.05.txt",
    "Int_Mono_conditional_eQTL_FDR0.05.txt",
    "LDG_conditional_eQTL_FDR0.05.txt",
    "mDC_conditional_eQTL_FDR0.05.txt",
    "Mem_CD4_conditional_eQTL_FDR0.05.txt",
    "Mem_CD8_conditional_eQTL_FDR0.05.txt",
    "Naive_B_conditional_eQTL_FDR0.05.txt",
    "Naive_CD4_conditional_eQTL_FDR0.05.txt",
    "Naive_CD8_conditional_eQTL_FDR0.05.txt",
    "NC_Mono_conditional_eQTL_FDR0.05.txt",
    "Neu_conditional_eQTL_FDR0.05.txt",
    "NK_conditional_eQTL_FDR0.05.txt",
    "pDC_conditional_eQTL_FDR0.05.txt",
    "Plasmablast_conditional_eQTL_FDR0.05.txt",
    "SM_B_conditional_eQTL_FDR0.05.txt",
    "TEMRA_CD8_conditional_eQTL_FDR0.05.txt",
    "Tfh_conditional_eQTL_FDR0.05.txt",
    "Th17_conditional_eQTL_FDR0.05.txt",
    "Th1_conditional_eQTL_FDR0.05.txt",
    "Th2_conditional_eQTL_FDR0.05.txt",
    "USM_B_conditional_eQTL_FDR0.05.txt"
]

# Loop through the file names and process each file
for file_name in file_names:
    input_file = file_name
    output_file = file_name.replace(".txt", ".vcf")

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            fields = line.strip().split("\t")
            # Assuming that you want to keep columns 1, 2, 6, 7, 8, and 11
            selected_fields = [fields[i] for i in [0, 1, 5, 6, 7, 10]]
            outfile.write("\t".join(selected_fields) + "\n")

print("Conversion completed.")
