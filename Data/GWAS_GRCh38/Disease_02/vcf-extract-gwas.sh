#!/bin/bash

vcftools --gzvcf /mnt/data0/public_data/human_9606_b151_GRCh38p7/00-All.vcf.gz --out Disease_02-filter-SNP --snps Disease_sort_uniq_02_list --recode


