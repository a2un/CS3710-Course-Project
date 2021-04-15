from os import path, walk, mkdir
import pandas as pd
from sklearn.model_selection import train_test_split

def classlabel(entry):
    return 0 if entry == 'Argumentative' else 1


def create_dataset(source_path = './dataset/webis-debate-16/', dest_path = './dataset'):
    lines = []
    for cur_path,_,files in walk(source_path):
        for fname in files:
            lines.extend(open(path.join(cur_path,fname),'r',encoding='utf-8').readlines())
    
    
    templist = []

    for line in lines:
        entry = line.split('\t')

        templist.append([f'{classlabel(entry[0])}','',f'{entry[1]}',''])
    
#     print(templist)
    
    df = pd.DataFrame(templist,columns=['class','argtype','text','argtypetext'])
    
    df.to_csv(open(path.join(dest_path,'all_data.csv'),'w+',encoding='utf-8',errors='ignore'))

    train, test = train_test_split(df, test_size=0.2)
    
    train.to_csv(open(path.join(dest_path,'train.csv'),'w+',encoding='utf-8',errors='ignore'),index=False)
    test.to_csv(open(path.join(dest_path,'test.csv'),'w+',encoding='utf-8',errors='ignore'),index=False)