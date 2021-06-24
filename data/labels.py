#coding:utf-8
import json
import os
import numpy as np

#EAST label
def json2txt_EAST(path_json,path_txt):
    with open(path_json,'r') as path_json:
        jsonx=json.load(path_json)
        with open(path_txt,'w+') as ftxt:
            for shape in jsonx['shapes']:           
                xy=np.array(shape['points'])
                label=str(shape['label'])
                strxy = ''
                for m,n in xy:
                    m=int(m)
                    n=int(n)
                    strxy+=str(m)+','+str(n)+','
                strxy+=label
                print(strxy)                                             
                ftxt.writelines(strxy+"\n")   
#CRNN label
def json2txt_CRNN(path_json,path_txt):
    with open(path_json,'r') as path_json:
        jsonx=json.load(path_json)
        with open(path_txt,'w+') as ftxt:
            for shape in jsonx['shapes']:           
                label=str(shape['label'])  
                strxy = os.path.split(path_txt)[1].replace('txt','jpg')+" "+label
                print(strxy)                                             
                ftxt.writelines(strxy+"\n")  

def east_label(dir_json,dir_txt):
    list_json = os.listdir(dir_json)
    for cnt,json_name in enumerate(list_json):
        print(f'cnt= {cnt},name= {json_name}')
        path_json =os.path.join(dir_json,json_name)
        path_txt = os.path.join(dir_txt,json_name.replace('.json','.txt'))
        #print(path_json,path_txt)    
        json2txt_EAST(path_json,path_txt)

def crnn_label(dir_json,dir_txt):
    list_json = os.listdir(dir_json)
    for cnt,json_name in enumerate(list_json):
        print(f'cnt= {cnt},name= {json_name}')
        path_json = os.path.join(dir_json,json_name)
        path_txt = os.path.join(dir_txt,json_name.replace('.json','.txt'))
        #print(path_json,path_txt)    
        json2txt_CRNN(path_json,path_txt)

if __name__ == "__main__":
    root_path = r'E:\Code\Python\datas\meter\DMkinds'
    dir_json = os.path.join(root_path,r'DM1json')
    dir_east_txt = os.path.join(root_path,r'DM1east')
    dir_crnn_txt = os.path.join(root_path,r'DM1crnn')
    east_label(dir_json,dir_east_txt)
    crnn_label(dir_json,dir_crnn_txt)
