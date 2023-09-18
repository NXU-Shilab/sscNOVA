import pandas as pd

left=pd.read_csv('Disease_01_chr_pos_sort_uniq_dbsnp_tabix.vcf', sep='\t')
right=pd.read_csv('Disease_sort_uniq_01_edit.vcf', sep='\t')
data_merge=pd.merge(left, right, on=['chr','pos'])
data_merge=data_merge.groupby(['chr','pos','ID','REF','ALT'])
data_merge=data_merge['DISEASE/TRAIT'].unique()
data_merge=data_merge.reset_index()
data_merge['DISEASE/TRAIT']=data_merge['DISEASE/TRAIT'].apply(lambda x: ';'.join(x))
data_merge.to_csv('Disease_01_merge_dbsnp.vcf', sep='\t', header=None, index=None)
