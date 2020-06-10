"""
Huajie Shao@2019/11/29
Fun: compute metrics:bleu1, bleu2, bleu3
"""

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import meteor_score,single_meteor_score
# from rouge import Rouge
from rouge import FilesRouge
from nltk import ngrams
from nltk import word_tokenize
import time
import multiprocessing
from functools import partial
import numpy as np
import os
import warnings
warnings.simplefilter("ignore")

value = 1000



def _compute_corpus_bleu(i,reference,predict):
    """compute corpus bleu
    input:list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    """
    # reference = [[ref.split()] for ref in reference]
    # predict = [pr.split() for pr in predict]
    # assert len(reference) == len(predict)
    
    # print("k process>>> ", i)
    # try:
        # bleu1 = corpus_bleu(reference[value*i:value*(i+1)], predict[value*i:value*(i+1)], weights=[1,0])
        # bleu2 = corpus_bleu(reference[value*i:value*(i+1)], predict[value*i:value*(i+1)],weights=[0.5,0.5])
    bleu1 = corpus_bleu(reference[1:3], predict[1:3], weights=[1,0])
    bleu2 = corpus_bleu(reference[1:3], predict[1:3], weights=[0.5,0.5])
    # print(bleu1,bleu2)
    # except:
    #   print("--bad case--"*10)
    #   bleu1 = 0.1
    #   bleu2 = 0.0017
    # bleu3 = corpus_bleu(reference, predict,weights=[1/3.0, 1/3.0, 1.0/3])
    # print("**compute bleu4 now...")
    # bleu4 = corpus_bleu(reference, predict)
    # print(bleu1,bleu2)
    return [bleu1,bleu2]
    

def _compute_sentence_bleu(k,reference,predict):
    """compute corpus bleu
    input:list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    """
    # reference = [[ref.split()] for ref in reference]
    # predict = [pr.split() for pr in predict]
    # assert len(reference) == len(predict)
    # print("sentence level")
    # print(reference[k])
    # print("predict:\n", predict[k])

    bleu1 = sentence_bleu([reference[k]], predict[k], weights=[1,0])
    bleu2 = sentence_bleu([reference[k]], predict[k], weights=[0.5,0.5])
    bleu3 = sentence_bleu([reference[k]], predict[k], weights=[0.3333,0.33333,0.3333])
    # bleu4 = sentence_bleu([reference[k]], predict[k], weights=[0.25,0.25,0.25,0.25])
    bleu4 = 0.0
    # print(bleu1,bleu2)

    return [bleu1,bleu2,bleu3,bleu4]



def _read_text_file(fileName):
    text = []
    with open(fileName) as f:
        for num,line in enumerate(f):
            arr = line.split()
            text.append(arr)
    return text



def _multiple_run(file_ground,file_predict):
    gound = _read_text_file(file_ground)
    predict = _read_text_file(file_predict)
    # print ("ground:\n", gound)
    # print("predict:\n", predict)
    N = len(predict)
    k_list = np.arange(0,N)
    print("----total processes>>>>: ", len(k_list))
    
    pool = multiprocessing.Pool(processes=48)
    fun = partial(_compute_sentence_bleu, reference=gound,predict=predict)
    result = pool.map(fun, k_list)
    result = np.mean(result,axis=0)
    bleu1 = result[0]
    bleu2 = result[1]
    bleu3 = result[2]
    # bleu4 = result[3]
    bleu4 = 0
    
    return bleu1,bleu2,bleu3,bleu4
    

def main():
    N = 6
    path_list = ['cost_anneal_20000-v', 'cyclical_4-v', 'pid25-v','pid35-v']
    with open('./result/BLEU.txt' , "w") as fout:
        for path in path_list:
            print("path: ", path)
            bleu1_list = []
            bleu2_list = []
            bleu3_list = []
            bleu4_list = []
            for i in range(1,N):
                path_name = path+str(i)
                file_ground = os.path.join(path_name, 'ground.txt')
                file_predict = os.path.join(path_name, 'generated.txt')
                bleu1,bleu2,bleu3,bleu4 = _multiple_run(file_ground, file_predict)
                bleu1_list.append(bleu1)
                bleu2_list.append(bleu2)
                bleu3_list.append(bleu3)
                # bleu4_list.append(bleu4)
            ## write result to file
            bleu1_mean = np.mean(bleu1_list)
            bleu1_var = np.std(bleu1_list)
            bleu2_mean = np.mean(bleu2_list)
            bleu2_var = np.std(bleu2_list)
            bleu3_mean = np.mean(bleu3_list)
            bleu3_var = np.std(bleu3_list)

            fout.write(path + " bleu-1 mean: {:4f} var: {:.4f} bleu-2 mean: {:4f} var: {:3f} bleu-3 mean: {:4f} var: {:4f}\n"\
                    .format(bleu1_mean, bleu1_var, bleu2_mean, bleu2_var, bleu3_mean, bleu3_var))
            fout.flush()
            

## --main function--
if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time", (time_end-time_start)/60)
    


