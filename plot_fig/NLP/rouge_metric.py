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
from rouge import FilesRouge
import json
import os


def _compute_corpus_bleu(reference,predict):
	"""compute corpus bleu
	input:list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
	"""
	# reference = [[ref.split()] for ref in reference]
	# predict = [pr.split() for pr in predict]
	# assert len(reference) == len(predict)
	bleu1 = corpus_bleu(reference, predict,weights=[1,0])
	print("**compute bleu2 now...")
	bleu2 = corpus_bleu(reference, predict,weights=[0.5,0.5])
	# bleu3 = corpus_bleu(reference, predict,weights=[1/3.0, 1/3.0, 1.0/3])
	# print("**compute bleu4 now...")
	# bleu4 = corpus_bleu(reference, predict)
	
	return bleu1,bleu2,bleu3,bleu4


def _compute_sentence_bleu(reference,predict):
	"""compute corpus bleu
	input:list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
	"""
	# reference = [[ref.split()] for ref in reference]
	# predict = [pr.split() for pr in predict]
	# assert len(reference) == len(predict)
	print(reference)
	print(predict)
	bleu1 = sentence_bleu(reference, predict,weights=[1,0])
	print("**compute bleu2 now...")
	bleu2 = sentence_bleu(reference, predict,weights=[0.5,0.5])
	bleu3 = sentence_bleu(reference, predict,weights=[1/3.0, 1/3.0, 1.0/3])
	print("**compute bleu4 now...")
	bleu4 = sentence_bleu(reference, predict)
	
	return bleu1,bleu2,bleu3,bleu4


def _compute_meteor(reference,predict):
	"""Fun:compute meteor score
		Input: sentence with string, e.g: reference= "I have a car"
		For meteor_score: [reference1, reference2, reference3]
		input: predict is the string of sentence
	"""
	# meteor = meteor_score(reference,predict)
	#### single mean one to one
	meteor = single_meteor_score(reference,predict)
	return round(meteor,4)



def _compute_sent_rouge(reference,predict):
	"""compute rouge metric
	Input: a sentence of string, for example:reference="I have a car"
	"""
	rouge = Rouge()
	scores = rouge.get_scores(predict, reference, avg=True)

	return scores


def _compute_file_rouge(ref_path,hyp_path):
	"""hyp_path:predict file"""
	files_rouge = FilesRouge(hyp_path,ref_path)
	scores = files_rouge.get_scores(avg=True)
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
			arr = line.split()
			text.append(arr)
	return text


def main():
	path_list = ['cost_anneal_20000-v', 'cyclical_4-v', 'pid17-v', 'pid25-v','pid35-v']
	with open('./result/rouge.txt' , "w") as fout:
		for path in path_list:
			res_list = []
			for i in [1,2,3,4]:
				path_name = path+str(i)
				file_ground = os.path.join(path_name, 'ground.txt')
				file_predict = os.path.join(path_name, 'generated.txt')
				scores = _compute_file_rouge(file_ground, file_predict)
				Rouge_L = scores['rouge-l']
			## write result to file
			print(Rouge_L)
			fout.write(path + "\t" + str(Rouge_L))
			fout.write("\n")
			

if __name__ == '__main__':
	time_start = time.time()
	main()
	time_end = time.time()
	print("running time: ", time_end - time_start)
	




