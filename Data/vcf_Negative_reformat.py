file = open("Negative.vcf")

lines = file.readlines()
for line in lines:
    line = line.rstrip('\n')
    line_split = line.split('\t')
    chr = line_split[0]
    pos = line_split[1]
    rsid = "."
    ref = line_split[3]
    alt = line_split[4]
    print(chr+'\t'+pos+'\t'+rsid+'\t'+ref+'\t'+alt)
