#!/bin/bash

cat Disease_00_filter-SNP.recode.dbsnp.vcf | while read line
do
    eval $(echo "$line" | awk -F '\t' '{printf("CHR=%s\nPOS=%s",$1,$2)}')
    COL=":"
    SEP="-"
    line="$CHR$COL$POS$SEP$POS"
    echo "$line" >> Disease_00_filter-SNP.recode.dbsnp_chr_pos.vcf
        
done
