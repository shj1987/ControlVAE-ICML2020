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
import os
import numpy as np



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

	meteor = meteor_score(reference,predict)
	#### single mean one to one
	# meteor = single_meteor_score(reference,predict)
	return meteor



def _compute_rouge(reference,predict):
	"""compute rouge metric
	Input: a sentence of string, for example:reference="I have a car"
	"""
	rouge = Rouge()
	scores = rouge.get_scores(predict, reference, avg=True)

	return scores


def _token_count(predict):
	"""
	Fun:the total number of words in the predict/generated text
	"""
	words_list = []
	for sen in predict:
		words_list.extend(sen)
		
	return len(words_list)


def _distinct_n_gram(predict,n=2):
	"""
	Fun:compute the distinct number of unigrams, bigrams and trigrams
	input: n-gram
	return: number of distinct n-grams
	"""

	predict_list = []
	for l in predict:
		predict_list.extend(l)

	n_dis = ngrams(predict_list, n)
	n_dis = set(n_dis)
	
	return len(n_dis)


def _read_text_file(fileName):
	predict_list = []
	target_list = []
	predict_sent = []
	with open(fileName) as f:
		for num,line in enumerate(f):
			if (num+1) % 14 == 3:
				target = line.split('>>')[1]
			if 4 <= (num+1) % 14 <= 13:
				predict = line.split('>>')[1]
				predict_sent.append(predict)

				arr = predict.split()
				predict_list.append(arr)
				target_list.append(target)

	assert len(predict_sent) == len(target_list)
	## write ground truth and predict to files[
	folder = fileName.split('/')[0]
	with open(folder+"/ground.txt", "w") as f_ground:
		with open(folder+"/generated.txt", "w") as f_predict:
			for k in range(len(target_list)):
				ground = target_list[k]
				predict = predict_sent[k]
				f_ground.write(ground)
				f_predict.write(predict)

	return predict_list
	

def main():
	if not os.path.exists('result'):
		os.makedirs('result')

	## read the PID-KL loss
	fw = open('result/Dis_mean_var.txt' , "w")
	path_list = ['cost_anneal_20000-v', 'cyclical_4-v', 'pid17-v', 'pid25-v','pid35-v']
	for path in path_list:
		dis1_list = []
		dis2_list = []
		dis3_list = []
		dis4_list = []
		for i in [1,2,3,4]:
			path_name = path+str(i)
			fileName = os.path.join(path_name, 'test_'+str(i))
			print("filename >>", fileName)
			predicts = _read_text_file(fileName)
			dis1 = _distinct_n_gram(predicts,1)
			dis2 = _distinct_n_gram(predicts,2)
			dis3 = _distinct_n_gram(predicts,3)
			dis4 = _distinct_n_gram(predicts,4)
			dis1_list.append(dis1)
			dis2_list.append(dis2)
			dis3_list.append(dis3)
			dis4_list.append(dis4)
		print(path + " >> MEAN distinct 1, 2, 3, 4:>>  ", np.mean(dis1_list),\
				np.mean(dis2_list), np.mean(dis3_list), np.mean(dis4_list))
		print(path + " >> Std distinct 1, 2, 3, 4:>> ", np.std(dis1_list),\
				np.std(dis2_list), np.std(dis3_list), np.std(dis4_list))
		fw.write(path + " >> MEAN distinct 1, 2, 3, 4:>> {:.1f}, {:.1f}, {:.1f}, {:.1f}\n".format(np.mean(dis1_list),\
				np.mean(dis2_list), np.mean(dis3_list), np.mean(dis4_list)))
		fw.write(path + " >> Standard distinct 1, 2, 3, 4:>> {:.1f}, {:.1f}, {:.1f}, {:.1f}\n".format(np.std(dis1_list),\
				np.std(dis2_list), np.std(dis3_list), np.std(dis4_list)))
		fw.flush()
		print('-'*80)
		# break
	fw.close()



if __name__ == '__main__':
	time_start = time.time()
	main()
	time_end = time.time()
	print("running time: ", time_end - time_start)
	




