import re

file = open("Disease_merge.vcf")
lines = file.readlines()
for line in lines:
    line = line.rstrip('\n')
    find_result = re.search('#', line)
    if find_result:
        continue
    else:
        line_split = line.split('\t')
        chr = line_split[0]
        pos = line_split[1]
        rsid = line_split[2]
        ref = line_split[3]
        alt = line_split[4]
        info = line_split[5]

        find_result_comma_ref = re.search(',', ref)
        find_result_comma_alt = re.search(',', alt)

        if find_result_comma_ref:
            ref_split = ref.split(',')
            if find_result_comma_alt:
                alt_split = alt.split(',')
                for ref_index in range(len(ref_split)):
                    for alt_index in range(len(alt_split)):
                        print("chr"+chr+'\t'+pos+'\t'+rsid+'\t'+ref_split[ref_index]+'\t'+alt_split[alt_index]+'\t'+info)
            else:
                for ref_index in range(len(ref_split)):
                    print("chr"+chr+'\t'+pos+'\t'+rsid+'\t'+ref_split[ref_index]+'\t'+alt+'\t'+info)
        elif find_result_comma_alt:
            alt_split = alt.split(',')
            for alt_index in range(len(alt_split)):
                print("chr"+chr+'\t'+pos+'\t'+rsid+'\t'+ref+'\t'+alt_split[alt_index]+'\t'+info)
        else:
            print("chr"+chr+'\t'+pos+'\t'+rsid+'\t'+ref+'\t'+alt+'\t'+info)
