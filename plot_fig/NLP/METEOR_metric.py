"""
Huajie Shao@2019/11/29
Fun: compute metrics:bleu1, bleu2, bleu3
"""

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score,single_meteor_score
from rouge import Rouge
from rouge import FilesRouge
from nltk import ngrams
from nltk import word_tokenize
import time
import multiprocessing
from functools import partial
import numpy as np
import os
# import nltk
# nltk.download('wordnet')


def _compute_meteor(k,reference,predict):
	"""Fun:compute meteor score
		Input: sentence with string, e.g: reference= "I have a car"
		For meteor_score: [reference1, reference2, reference3]
		input: predict is the string of sentence
	"""
	# meteor = meteor_score(reference,predict)
	#### single mean one to one
	meteor = single_meteor_score(reference[k],predict[k])
	# print("current K: ", k)
	return round(meteor,4)
	

def _compute_rouge(reference,predict):
	"""compute rouge metric
	Input: a sentence of string, for example:reference="I have a car"
	"""
	rouge = Rouge()
	scores = rouge.get_scores(predict, reference, avg=True)

	return scores
	

def _distinct_n_gram(predict,n=2):
	"""
	Fun:compute the distinct number of unigrams, bigrams and trigrams
	input: n-gram
	return: number of distinct n-grams
	"""
	# predict = predict.lower()
	predict_list = []
	for l in predict:
		predict_list.extend(l)

	n_dis = ngrams(predict_list, n)
	n_dis = set(n_dis)
	return len(n_dis)


def _read_text_file(fileName):
	text = []
	with open(fileName) as f:
		for num,line in enumerate(f):
			line = line.replace('<unk> ',"")
			text.append(line)

	return text
	

def _multiple_run(file_ground, file_predict):
	gound = _read_text_file(file_ground)
	predict = _read_text_file(file_predict)
	
	k_list = np.arange(0,len(predict))
	# k_list = np.arange(0,10)
	print("number of multiprocessing K: >>>",len(k_list))
	
	pool = multiprocessing.Pool(processes=10)
	fun = partial( _compute_meteor, reference=gound,predict=predict)
	result = pool.map(fun, k_list)
	## get result
	meteor = 0
	for i in range(len(result)):
		meteor += result[i]

	meteor = meteor/len(result)
	return meteor
	

def main():

	meteor_list = []
	path_list = ['cost_anneal_20000-v', 'cyclical_4-v', 'pid17-v', 'pid25-v','pid35-v']
	with open('./result.txt' , "w") as fout:
		for path in path_list:
			res_list = []
			for i in [1,2,3,4]:
				path_name = path+str(i)
				file_ground = os.path.join(path_name, 'ground.txt')
				file_predict = os.path.join(path_name, 'generated.txt')
				meteor = _multiple_run(file_ground, file_predict)
				res_list.append(meteor)
			## get the average
			avg = np.mean(res_list)
			var = np.std(res_list)
			fout.write(path_name + "mean: {:.4f}, var{:.4f} \n".format(avg,var))
			

if __name__ == '__main__':
	time_start = time.time()
	main()
	time_end = time.time()
	print("running time: ", time_end - time_start)
	


