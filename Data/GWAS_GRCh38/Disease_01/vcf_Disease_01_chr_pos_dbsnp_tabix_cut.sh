#!/bin/bash

cat Disease_01_chr_pos_sort_uniq_dbsnp_tabix.vcf | while read line
do
    eval $(echo "$line" | awk -F '\t' '{printf("CHR=%s\nPOS=%s",$1,$2)}')
    COL=":"
    SEP="-"
    line="$CHR$COL$POS$SEP$POS"
    echo "$line" >> Disease_01_chr_pos_sort_uniq_dbsnp_tabix_cut.vcf
        
done

