import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from Metrics import Metrics
import time


class SelfBleu(Metrics):
    def __init__(self, test_text='', gram=3):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 10000
        self.reference = None
        self.is_first = True
        
    def get_name(self):
        return self.name
        
    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def _read_text_file(self, fileName):
        text = []
        with open(fileName) as f:
            for num,line in enumerate(f):
                arr = line.split()
                text.append(arr)
        return text
    
    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as real_data:
                for text in real_data:
                    text = text.replace('<unk> ', "")
                    # text = nltk.word_tokenize(text)
                    text = text.split()
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = hypothesis.split()
                # hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)
        
    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        print("cpu number: ", os.cpu_count())
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))
            
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt
        

def main():
    # path = './files/File_bleu/'
    path = './files/'
    file_list = ['test.cvae.512.kl_5.output_sample','test.cvae.512.kl_2.output_sample','test.cvae.512.kl_1.output_sample',\
                'test.cvae.512.cost_annealing.output_sample','test.cvae.512.cyclical.output_sample',\
                'expansion_tst_sample.txt']
    ## get file list
    methods = ['KL-5', 'KL-2', 'KL-1', 'cost_annealing', 'cyclical', 'expansionNet']
    with open('self_bleu.txt', "w") as fw:
        for num, file in enumerate(file_list):
            print("filename: >>", file)
            test_text = os.path.join(path,file)
            gram = 2
            self_bleu = SelfBleu(test_text, gram)
            bleu_2 = self_bleu.get_score()
            ## compute bleu 2
            gram = 3
            self_bleu = SelfBleu(test_text, gram)
            bleu_3 = self_bleu.get_score()
            print(methods[num] + " >> bleu-2: {:.4f} bleu-3: {:.4f}\n".format(bleu_2, bleu_3))
            out_res = methods[num] + " >> bleu-2: {:.4f} bleu-3: {:.4f}\n".format(bleu_2, bleu_3)
            fw.write(out_res)
            fw.flush()
            

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('running time:', (time_end - time_start)/60.0, 'mins')
    



