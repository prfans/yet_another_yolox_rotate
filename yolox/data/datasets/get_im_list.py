# coding: gbk
'''
get image list
'''

import os

def get_im__(path,ext):
    if '.' in ext:
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]
    else:
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.' + ext)]

def get_im_list(filepath,ext):
    #filepath=filepath.encode('gbk')
    files_list=[]
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        #print fi_d
        #print fi_d
        if os.path.isdir(fi_d):
            #files_list += get_im__(fi_d, ext)
            #print fi_d
            files_list += get_im_list(fi_d,ext)
        else:
            if fi_d.endswith(ext):
                files_list.append(os.path.join(filepath,fi))

    return  files_list



#def get_im_list(filepath,ext):
#    return [i for i in os.listdir('.') if os.path.isfile(i) and os.path.splitext(i)[1]==ext]