import subprocess

sample_names = [
    "CD16p_Mono",
    "CL_Mono",
    "CM_CD8",
    "DN_B",
    "EM_CD8",
    "Fr_I_nTreg",
    "Fr_II_eTreg",
    "Fr_III_T",
    "Int_Mono",
    "LDG",
    "mDC",
    "Mem_CD4",
    "Mem_CD8",
    "Naive_B",
    "Naive_CD4",
    "Naive_CD8",
    "NC_Mono",
    "Neu",
    "NK",
    "pDC",
    "Plasmablast",
    "SM_B",
    "TEMRA_CD8",
    "Tfh",
    "Th17",
    "Th1",
    "Th2",
    "USM_B",
]

for sample_name in sample_names:
    input_file = f"{sample_name}_conditional_eQTL_FDR0.05_SNPS_list_filter_SNP.recode.vcf"
    output_file = f"{sample_name}.vcf"

    # Construct the awk and tr commands
    awk_command = f'awk -F "\\t" \'{{if ($8 == ".") $8 = "{sample_name}"}}1\' {input_file}'
    tr_command = f'tr " " "\\t"'

    # Execute the commands using subprocess
    awk_process = subprocess.Popen(awk_command, shell=True, stdout=subprocess.PIPE)
    tr_process = subprocess.Popen(tr_command, shell=True, stdin=awk_process.stdout, stdout=subprocess.PIPE)

    # Redirect the output to the output file
    with open(output_file, "wb") as output:
        output.write(tr_process.communicate()[0])

print("Processing completed.")
