#!/bin/bash

cat Disease_sort_uniq_01_cut.vcf | while read line
do
    eval $(echo "$line" | awk -F ':' '{printf("CHR=%s\nPOS=%s",$1,$2)}')
    COL=":"
    SEP="-"
    line="$CHR$COL$POS$SEP$POS"
    echo "$line" >> Disease_01_chr_pos.vcf
        
done
