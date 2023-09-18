# List of input file names
input_files = [
    "CD16p_Mono.vcf",
    "CL_Mono.vcf",
    "CM_CD8.vcf",
    "DN_B.vcf",
    "EM_CD8.vcf",
    "Fr_II_eTreg.vcf",
    "Fr_III_T.vcf",
    "Fr_I_nTreg.vcf",
    "Int_Mono.vcf",
    "LDG.vcf",
    "mDC.vcf",
    "Mem_CD4.vcf",
    "Mem_CD8.vcf",
    "Naive_B.vcf",
    "Naive_CD4.vcf",
    "Naive_CD8.vcf",
    "NC_Mono.vcf",
    "Neu.vcf",
    "NK.vcf",
    "pDC.vcf",
    "Plasmablast.vcf",
    "SM_B.vcf",
    "TEMRA_CD8.vcf",
    "Tfh.vcf",
    "Th17.vcf",
    "Th1.vcf",
    "Th2.vcf",
    "USM_B.vcf",
]

output_file = "ImmuNexUT.vcf"

# Open the output file for appending
with open(output_file, "a") as output:
    # Iterate through each input file
    for input_file in input_files:
        with open(input_file, "r") as input:
            # Skip the first line
            next(input)
            
            # Append the rest of the lines to the output file
            for line in input:
                output.write(line)

print("Files appended to ImmuNexUT.vcf.")
