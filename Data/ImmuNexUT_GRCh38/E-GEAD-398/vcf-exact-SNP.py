import os

# Define a list of file names
file_names = [
    "CD16p_Mono_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "CL_Mono_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "CM_CD8_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "DN_B_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "EM_CD8_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Fr_I_nTreg_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Fr_II_eTreg_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Fr_III_T_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Int_Mono_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "LDG_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "mDC_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Mem_CD4_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Mem_CD8_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Naive_B_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Naive_CD4_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Naive_CD8_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "NC_Mono_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Neu_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "NK_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "pDC_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Plasmablast_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "SM_B_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "TEMRA_CD8_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Tfh_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Th17_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Th1_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "Th2_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf",
    "USM_B_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf"
]

# Loop through the file names and process each file
for file_name in file_names:
    # Remove "_Forward_nominal_P" from the file name
    base_name = os.path.splitext(file_name)[0]  # Remove file extension
    output_file = base_name.replace("_Forward_nominal_P", "") + ".vcf_SNPS_list"

    with open(file_name, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            fields = line.strip().split("\t")
            snp_id = fields[2]
            outfile.write(snp_id + "\n")

print("Processing completed.")
