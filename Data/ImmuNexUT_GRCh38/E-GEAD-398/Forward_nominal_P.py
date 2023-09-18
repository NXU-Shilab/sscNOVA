import os

# Define a list of file names
file_names = [
    "CD16p_Mono_conditional_eQTL_FDR0.05.vcf",
    "CL_Mono_conditional_eQTL_FDR0.05.vcf",
    "CM_CD8_conditional_eQTL_FDR0.05.vcf",
    "DN_B_conditional_eQTL_FDR0.05.vcf",
    "EM_CD8_conditional_eQTL_FDR0.05.vcf",
    "Fr_I_nTreg_conditional_eQTL_FDR0.05.vcf",
    "Fr_II_eTreg_conditional_eQTL_FDR0.05.vcf",
    "Fr_III_T_conditional_eQTL_FDR0.05.vcf",
    "Int_Mono_conditional_eQTL_FDR0.05.vcf",
    "LDG_conditional_eQTL_FDR0.05.vcf",
    "mDC_conditional_eQTL_FDR0.05.vcf",
    "Mem_CD4_conditional_eQTL_FDR0.05.vcf",
    "Mem_CD8_conditional_eQTL_FDR0.05.vcf",
    "Naive_B_conditional_eQTL_FDR0.05.vcf",
    "Naive_CD4_conditional_eQTL_FDR0.05.vcf",
    "Naive_CD8_conditional_eQTL_FDR0.05.vcf",
    "NC_Mono_conditional_eQTL_FDR0.05.vcf",
    "Neu_conditional_eQTL_FDR0.05.vcf",
    "NK_conditional_eQTL_FDR0.05.vcf",
    "pDC_conditional_eQTL_FDR0.05.vcf",
    "Plasmablast_conditional_eQTL_FDR0.05.vcf",
    "SM_B_conditional_eQTL_FDR0.05.vcf",
    "TEMRA_CD8_conditional_eQTL_FDR0.05.vcf",
    "Tfh_conditional_eQTL_FDR0.05.vcf",
    "Th17_conditional_eQTL_FDR0.05.vcf",
    "Th1_conditional_eQTL_FDR0.05.vcf",
    "Th2_conditional_eQTL_FDR0.05.vcf",
    "USM_B_conditional_eQTL_FDR0.05.vcf"
]

# Loop through the file names and process each file
for file_name in file_names:
    output_file = file_name.replace(".vcf", "_Forward_nominal_P.vcf")

    with open(file_name, "r") as infile, open(output_file, "w") as outfile:
        first_line = infile.readline()  # Read the header line
        outfile.write(first_line)  # Write the header line to the output file

        for line in infile:
            fields = line.strip().split("\t")
            # Assuming that the Forward_nominal_P value is in the 6th column
            if float(fields[5]) < 0.05:
                outfile.write(line)
                
print("Processing completed.")
