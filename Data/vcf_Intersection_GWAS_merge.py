import pandas as pd

left=pd.read_csv('Intersection.vcf', sep='\t')
right=pd.read_csv('GWAS.vcf', sep='\t')
data_merge=pd.merge(left, right, on=['chr','pos','REF','ALT'])
data_merge=data_merge.groupby(['chr','pos','REF','ALT'])
data_merge=data_merge['DISEASE/TRAIT'].unique()
data_merge=data_merge.reset_index()
data_merge['DISEASE/TRAIT']=data_merge['DISEASE/TRAIT'].apply(lambda x: ';'.join(x))
data_merge.to_csv('Intersection_GWAS_merge.vcf', sep='\t', index=None)
