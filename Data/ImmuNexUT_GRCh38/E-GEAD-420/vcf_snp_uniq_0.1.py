# List of input files and output files
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
    "USM_B.vcf"
]

for input_file in input_files:
    # Define output file name
    output_file = input_file.split('.')[0] + '.txt'

    # Read the lines from the input files and convert them into sets
    with open(input_file, 'r') as f1, open(f'{input_file.split(".")[0]}_2.vcf', 'r') as f2:
        lines1 = set(f1.readlines())
        lines2 = set(f2.readlines())

    # Find the lines that are unique to the first file (subtract the sets)
    unique_lines = lines1 - lines2

    # Write the unique lines to the output file
    with open(output_file, 'w') as output:
        output.writelines(sorted(unique_lines))
