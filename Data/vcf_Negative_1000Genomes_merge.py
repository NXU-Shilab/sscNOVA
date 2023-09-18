import pandas as pd

left=pd.read_csv('Negative.vcf', sep='\t')
right=pd.read_csv('../1000Genomes/1000Genomes.vcf', sep='\t')
data_merge=pd.merge(left, right, on=['chr','pos','ID','REF','ALT'])
data_merge.to_csv('Negative_1000Genomes_merge.vcf', sep='\t', header=None, index=None)
