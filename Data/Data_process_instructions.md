# Data process

### GWAS_Catalog data

Based on the diseases names, we extract the raw variants data for 31 immune-related diseases from the GWAS Catalog data 
(e.g. ```less gwas_catalog_v1.0-associations_e0_r2022-11-08.tsv | cut -f 8,12,13,22 | grep -i Allergy > Allergy.txt```).

Then, we convert the variants for each disease into VCF format using the 1000 Genomes and dbSNP databases. 

Finally, we merge the variants data for 31 diseases together and remove duplicates.

### ImmuNexUT data

##### E-GEAD-398

Based on the cell subsets names, we extract the variants data for 28 immune-related cell subsets from the ImmuNexUT E-GEAD-398 data 
(e.g. ```less CD16p_Mono_conditional_eQTL_FDR0.05.txt | cut -f 1,2,6,7,8,11 > CD16p_Mono_conditional_eQTL_FDR0.05.vcf```).

Then, we obtain variants data which Forward_nominal_P < 0.05 
(e.g. ```awk -F '\t' '$6 < 0.05' CD16p_Mono_conditional_eQTL_FDR0.05.vcf > CD16p_Mono_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf```). 

For each cell subset, we next obtain rsid 
(e.g. ```less CD16p_Mono_conditional_eQTL_FDR0.05_Forward_nominal_P.vcf | cut -f 3 > CD16p_Mono_conditional_eQTL_FDR0.05.vcf_SNPS_list```). 

Furthermore, we use vcftools extract variants data in VCF format based on the accession number E-GEAD-420 
(e.g. ```vcftools --gzvcf ../E-GEAD-420/Cell_Subset_nominal_sort_uniq.vcf.gz --out CD16p_Mono_conditional_eQTL_FDR0.05_SNPS_list_filter_SNP --snps CD16p_Mono_conditional_eQTL_FDR0.05.vcf_SNPS_list --recode```). 

Add additional information with the cell subsets names for each variant 
(e.g. ```awk -F '\t' '{if($8==".") $8="CD16p_Mono"}1' CD16p_Mono_conditional_eQTL_FDR0.05_SNPS_list_filter_SNP.recode.vcf | tr ' ' '\t' > CD16p_Mono.vcf```). 

Finally, we merge the variants data for 28 cell subsets together and remove duplicates.

##### E-GEAD-420

First of all, we extract variant data for 28 immune-related cell subsets from ImmuNexUT E-GEAD-420 data based on the cell subset names with nominal_P_value > 0.1 
(e.g. ```awk -F '\t' '$12 > 0.1' CD16p_Mono_nominal.txt | cut -f 6-10 > CD16p_Mono_nominal_P_value.txt```). 

Then, we filter variants based on allele frequency greater than 0.3 in the 1000 Genomes dataset. 

Finally, we merge the variants data for 28 cell subsets together and remove duplicates.

### HGMD & ClinVar data

we convert the 140 positive variants used by Yousefian-Jazi et al. into VCF format using the 1000 Genomes and dbSNP databases. 

Following that, we filter the variants located within 1kbp upstream and downstream of the chromosomal positions of the 118 positive variants. 

Finally, we compute the conservation scores for these variants and only keep the ones with a phastcons100way conservation score below 0.5 and an allele frequency greater than 0.3.


