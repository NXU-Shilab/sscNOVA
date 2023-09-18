import pandas as pd

left=pd.read_csv('Disease_02-filter-SNP.recode.dbsnp.vcf', sep='\t')
right=pd.read_csv('Disease_sort_uniq_02.vcf', sep='\t')
data_merge=pd.merge(left, right, on='ID')
data_merge=data_merge.groupby(['CHROM','POS','ID','REF','ALT'])
data_merge=data_merge['DISEASE/TRAIT'].unique()
data_merge=data_merge.reset_index()
data_merge['DISEASE/TRAIT']=data_merge['DISEASE/TRAIT'].apply(lambda x: ';'.join(x))
data_merge.to_csv('Disease_02_merge.vcf', sep='\t', header=None, index=None)
