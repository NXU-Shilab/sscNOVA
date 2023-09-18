#!/bin/bash

cat Disease_01_chr_pos_sort_uniq.vcf | while read line
do
    tabix /mnt/data0/public_data/human_9606_b151_GRCh38p7/00-All.vcf.gz $line >> Disease_01_chr_pos_sort_uniq_dbsnp_tabix.vcf
        
done

