file = open("Intersection_GWAS_ImmuNexUT_merge.vcf")

lines = file.readlines()
for line in lines:
    line = line.rstrip('\n')
    line_split = line.split('\t')
    chr = line_split[0]
    pos = line_split[1]
    rsid = "."
    ref = line_split[2]
    alt = line_split[3]
    disease_trait = line_split[4]
    cell_subset = line_split[5]
    print(chr+'\t'+pos+'\t'+rsid+'\t'+ref+'\t'+alt+'\t'+disease_trait+'\t'+cell_subset)
