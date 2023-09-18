import os

strA = "h"
strB = "rs"

# Process input lines and categorize them into output files based on conditions
def process_file(input_file, output_prefix):
    with open(input_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                CHR_ID, CHR_POS, SNPS = parts[1], parts[2], parts[3]
                if CHR_ID and CHR_POS and SNPS:
                    output_file = f"{output_prefix}.vcf"
                elif not CHR_ID and not CHR_POS and strA in SNPS:
                    output_file = f"{output_prefix}_01.vcf"
                elif not CHR_ID and not CHR_POS and strB in SNPS:
                    output_file = f"{output_prefix}_02.vcf"
                else:
                    output_file = f"{output_prefix}_03.vcf"
                with open(output_file, 'a') as outfile:
                    outfile.write(line)

# Process each input file separately
input_files = [
    "Allergy.txt",
    "Alopecia_areata.txt",
    "Ankylosing_spondylitis.txt",
    "Asthma.txt",
    "Autoimmune_thyroid_disease.txt",
    "Behcet.txt",
    "Celiac.txt",
    "Crohn.txt",
    "Gout.txt",
    "Graves.txt",
    "Idiopathic_inflammatory_myopathy.txt",
    "Immunoglobulin.txt",
    "Juvenile_idiopathic_arthritis.txt",
    "Kawasaki_disease.txt",
    "Multiple_sclerosis.txt",
    "Myasthenia_gravis.txt",
    "Narcolepsy.txt",
    "Osteoarthritis.txt",
    "Primary_biliary_cholangitis.txt",
    "Primary_sclerosing_cholangitis.txt",
    "Psoriasis.txt",
    "Rheumatoid_arthritis.txt",
    "Sarcoidosis.txt",
    "Sjogren.txt",
    "Systemic_lupus_erythematosus.txt",
    "Systemic_sclerosis.txt",
    "Type_1_diabetes.txt",
    "Ulcerative_colitis.txt",
    "Vasculitis.txt",
    "Vitiligo.txt",
    "Takayasu_Arteritis.txt"
]

for input_file in input_files:
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    process_file(input_file, file_name)
