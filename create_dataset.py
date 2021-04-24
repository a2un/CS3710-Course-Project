from os import path, walk, mkdir
import pandas as pd
from sklearn.model_selection import train_test_split

def classlabel(entry,ttype='argumentative'):
    if ttype != 'argumentative':
        return 2 if entry == 'Premise' else 3 if entry == 'Major Claim' else 4
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


def construct_splits_essay(essays, dest_path):
    lines = [] 

    if not(path.exists(dest_path)):
        mkdir(dest_path)

    for essay in essays:
        essay_text = essay["text"]
        essay_ann = essay["ann"]    
        for line in essay_ann:
            elems = line.split('\t')
            if 'T' in elems[0]:
                entry = elems[1].split()[0]
                entry_start = int(elems[1].split()[1])
                entry_end = int(elems[1].split()[2])
                entry_text = elems[2].rstrip()
                lines.append([classlabel('argumentative'),classlabel(entry,'class'),essay_text[2].rstrip(),entry_text,entry_start,entry_end])
        
    df = pd.DataFrame(lines,columns=['class','argtype','text','argtypetext','argtypetextstart','argtypetextend']) 
    df.to_csv(open(path.join(dest_path,'all_data.csv'),'w+',encoding='utf-8',errors='ignore'))

    train, test = train_test_split(df, test_size=0.2)
    
    train.to_csv(open(path.join(dest_path,'train.csv'),'w+',encoding='utf-8',errors='ignore'),index=False)
    test.to_csv(open(path.join(dest_path,'test.csv'),'w+',encoding='utf-8',errors='ignore'),index=False)


def create_dataset_essay(source_path='./dataset/AAEC',dest_path='./dataset/AAEC-splits'):
    essays = []

    for curr_path,_,files in walk(source_path):
        for fname in files:
            if path.splitext(fname)[1] == '.ann':
                essay = {}
                essay["text"] = open(path.join(curr_path,'{0}.txt'.format(path.splitext(fname)[0])),'r').readlines()
                essay["ann"] = open(path.join(curr_path,'{0}.ann'.format(path.splitext(fname)[0])),'r').readlines()
                essays.append(essay)

    construct_splits_essay(essays,dest_path)

if __name__ == "__main__":
    create_dataset_essay()