import matplotlib.pyplot as plt
 
name_list = ['Asthma', 'Systemic lupus erythematosus', 'Rheumatoid arthritis', 'Crohn', 'Multiple sclerosis', 'Type 1 diabetes', \
            'Psoriasis', 'Ulcerative colitis', 'Ankylosing spondylitis', 'Osteoarthritis', 'Immunoglobulin', \
            'Primary sclerosing cholangitis', 'Gout', 'Celiac', 'Primary biliary cholangitis', 'Systemic sclerosis', \
            'Sarcoidosis', 'Takayasu Arteritis', 'Allergy', 'Autoimmune thyroid disease', 'Graves', 'Kawasaki disease', \
            'Vitiligo', 'Vasculitis', 'Juvenile idiopathic arthritis', 'Sjogren syndrome', 'Alopecia areata', \
            'Behcet', 'Narcolepsy', 'Myasthenia gravis', 'Idiopathic inflammatory myopathy']
num_list = [2986,1105,887,859,856,763,641,630,415,404,345,336,260,236,211,171,170,160,154,151,131,119,110,100,95,57,55,50,38,36,9]

fig, ax = plt.subplots(figsize=(16, 6))
rects=ax.bar(range(len(num_list)), num_list, color=['#FFD1D1', '#D1FFD1', '#D1D1FF', '#FFD1FF', '#D1FFFF', '#FFFFD1', '#E8ECF3', '#E5F3E5', \
                '#E1B8A6', '#E8E8E8', '#F5F5DC', '#FDFD96', '#F8E0B0', '#E6E6FA', '#F2F2F2', '#FFE4B5', \
                '#D2B48C', '#FFDAB9', '#E6E6FA', '#E0EEE0', '#F0FFF0', '#E0FFFF', '#FFFACD', '#FFE1FF', \
                '#FFF0F5', '#FFEFD5', '#FDF5E6', '#DCDCDC', '#00FF00', '#B0E0E6', '#E9967A'])

index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

plt.ylim(ymax=3500, ymin=0)
plt.xticks(index, name_list, size='small', rotation=88)
plt.xlabel("Autoimmune Disease Name")
plt.ylabel("Variant Count")

for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2 , height, str(height), ha='center', va='bottom')

plt.savefig('Disease_Num_GWAS.pdf', bbox_inches='tight', pad_inches=0.5)
