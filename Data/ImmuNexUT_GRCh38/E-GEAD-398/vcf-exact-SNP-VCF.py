import subprocess

# Define a list of sample names and corresponding SNP list file names
sample_names = [
    "CD16p_Mono_conditional_eQTL_FDR0.05",
    "CL_Mono_conditional_eQTL_FDR0.05",
    "CM_CD8_conditional_eQTL_FDR0.05",
    "DN_B_conditional_eQTL_FDR0.05",
    "EM_CD8_conditional_eQTL_FDR0.05",
    "Fr_I_nTreg_conditional_eQTL_FDR0.05",
    "Fr_II_eTreg_conditional_eQTL_FDR0.05",
    "Fr_III_T_conditional_eQTL_FDR0.05",
    "Int_Mono_conditional_eQTL_FDR0.05",
    "LDG_conditional_eQTL_FDR0.05",
    "mDC_conditional_eQTL_FDR0.05",
    "Mem_CD4_conditional_eQTL_FDR0.05",
    "Mem_CD8_conditional_eQTL_FDR0.05",
    "Naive_B_conditional_eQTL_FDR0.05",
    "Naive_CD4_conditional_eQTL_FDR0.05",
    "Naive_CD8_conditional_eQTL_FDR0.05",
    "NC_Mono_conditional_eQTL_FDR0.05",
    "Neu_conditional_eQTL_FDR0.05",
    "NK_conditional_eQTL_FDR0.05",
    "pDC_conditional_eQTL_FDR0.05",
    "Plasmablast_conditional_eQTL_FDR0.05",
    "SM_B_conditional_eQTL_FDR0.05",
    "TEMRA_CD8_conditional_eQTL_FDR0.05",
    "Tfh_conditional_eQTL_FDR0.05",
    "Th17_conditional_eQTL_FDR0.05",
    "Th1_conditional_eQTL_FDR0.05",
    "Th2_conditional_eQTL_FDR0.05",
    "USM_B_conditional_eQTL_FDR0.05",
]

# Define the path to the input VCF file
input_vcf = "../E-GEAD-420/Cell_Subset_nominal_sort_uniq.vcf.gz"

# Loop through the sample names and execute vcftools commands
for sample_name in sample_names:
    snp_list_file = f"{sample_name}.vcf_SNPS_list"
    output_prefix = f"{sample_name}_SNPS_list_filter_SNP"

    # Construct the vcftools command
    cmd = [
        "vcftools",
        "--gzvcf",
        input_vcf,
        "--out",
        output_prefix,
        "--snps",
        snp_list_file,
        "--recode",
    ]

    # Execute the vcftools command
    subprocess.run(cmd)

print("Processing completed.")
