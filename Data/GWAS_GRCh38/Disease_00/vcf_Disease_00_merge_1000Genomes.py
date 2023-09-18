import pandas as pd

left=pd.read_csv('Disease_00_filter-SNP.recode.1000Genomes.vcf', sep='\t')
right=pd.read_csv('Disease_sort_uniq_00_reduce.vcf', sep='\t')
data_merge=pd.merge(left, right, on=['chr','pos'])
data_merge=data_merge.groupby(['chr','pos','REF','ALT'])
data_merge=data_merge['DISEASE/TRAIT'].unique()
data_merge=data_merge.reset_index()
data_merge['DISEASE/TRAIT']=data_merge['DISEASE/TRAIT'].apply(lambda x: ';'.join(x))
data_merge.to_csv('Disease_00_merge_1000Genomes.vcf', sep='\t', header=None, index=None)
