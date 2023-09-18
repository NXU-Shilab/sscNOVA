#!/bin/bash

cat Disease_sort_uniq_01_dbsnp_others.vcf | while read line
do
    tabix /mnt/data0/public_data/ALL_WGS_1000Genomes/ALL.wgs.shapeit2_integrated_snvindels_v2a.GRCh38.27022019.sites.vcf.gz $line >> Disease_sort_uniq_01_dbsnp_others_tabix.vcf
        
done


