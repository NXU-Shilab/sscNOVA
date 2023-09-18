#!/bin/bash

cat Disease_sort_uniq_00_cut.vcf | while read line
do
    eval $(echo "$line" | awk -F '\t' '{printf("CHR=%s\nPOS=%s",$1,$2)}')
    COL=":"
    SEP="-"
    line="$CHR$COL$POS$SEP$POS"
    echo "$line" >> Disease_sort_uniq_00_chr_pos.vcf
        
done


