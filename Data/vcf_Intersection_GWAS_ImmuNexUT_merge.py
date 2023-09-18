import pandas as pd

left=pd.read_csv('Intersection_GWAS_merge.vcf', sep='\t')
right=pd.read_csv('ImmuNexUT.vcf', sep='\t')
data_merge=pd.merge(left, right, on=['chr','pos','REF','ALT'])
data_merge=data_merge.groupby(['chr','pos','REF','ALT','DISEASE/TRAIT'])
data_merge=data_merge['Cell_Subset'].unique()
data_merge=data_merge.reset_index()
data_merge['Cell_Subset']=data_merge['Cell_Subset'].apply(lambda x: ';'.join(x))
data_merge.to_csv('Intersection_GWAS_ImmuNexUT_merge.vcf', sep='\t', header=None, index=None)
