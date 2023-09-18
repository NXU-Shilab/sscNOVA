import matplotlib.pyplot as plt
 
name_list = ['Th1', 'CD16p_Mono', 'mDC', 'CL_Mono', 'SM_B', 'Tfh', 'Mem_CD4', 'Th17', 'EM_CD8', 'pDC', \
            'NK', 'Fr_II_eTreg', 'Mem_CD8', 'Neu', 'Naive_CD4', 'Th2', 'Naive_B', 'USM_B', 'Naive_CD8', 'DN_B', \
            'Fr_III_T', 'Int_Mono', 'NC_Mono', 'Plasmablast', 'TEMRA_CD8', 'CM_CD8', 'Fr_I_nTreg', 'LDG']
num_list = [1808,1730,1716,1707,1701,1623,1611,1596,1566,1565,1554,1511,1493,1476,1453,1441,1424,1414,1399,1381,1359,1340,1301,1293,1284,1112,1076,636]

fig, ax = plt.subplots(figsize=(16, 6))
rects=ax.bar(range(len(num_list)), num_list, color=['#FFD1D1', '#D1FFD1', '#D1D1FF', '#FFD1FF', '#D1FFFF', '#FFFFD1', '#E8ECF3', '#E5F3E5', \
                '#E1B8A6', '#E8E8E8', '#F5F5DC', '#FDFD96', '#F8E0B0', '#E6E6FA', '#F2F2F2', '#FFE4B5', \
                '#D2B48C', '#FFDAB9', '#E6E6FA', '#E0EEE0', '#F0FFF0', '#E0FFFF', '#FFFACD', '#FFE1FF', \
                '#FFF0F5', '#FFEFD5', '#FDF5E6', '#DCDCDC'])

index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

plt.ylim(ymax=2000, ymin=0)
plt.xticks(index, name_list, size='small', rotation=88)
plt.xlabel("Immune Cell Name")
plt.ylabel("Variant Count")

for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')

plt.savefig('Cell_Num_Positive.pdf', bbox_inches='tight', pad_inches=0.5)
