import pandas as pd

data=pd.read_csv('ImmuNexUT_sort.vcf', sep='\t')
data=data.groupby(['CHROM','POS','ID','REF','ALT'])
data=data['Cell_Subset'].unique()
data=data.reset_index()
data['Cell_Subset']=data['Cell_Subset'].apply(lambda x: ';'.join(x))
data.to_csv('ImmuNexUT_sort_merge.vcf', sep='\t', header=None, index=None)
