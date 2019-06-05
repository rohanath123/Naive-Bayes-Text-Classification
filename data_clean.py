import pandas as pd
import csv
import nltk
from nltk.tokenize.moses import MosesDetokenizer
nltk.download('words')
nltk.download('stopwords')

#USE WRITE_TO_EXCEL, PREPARE_DATA ONLY FOR CLEANING RAW DATASET. DONT USE THESE FUNCTIONS AS SUCH, IVE ALREADY INCLUDED THE CLEANED DATASET YOU NEED TO USE BELOW. 
def write_to_excel(data, mode):
    with open("C:/Users/Rohan/Desktop/Lakhan Model/cleaned.csv", mode, newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    f.close()

def prepare_data():
	words = list(nltk.corpus.words.words('en'))
	stopwords = list(nltk.corpus.stopwords.words('english'))
	spcl = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '=', '[', ']', '{', '}', '/', ':', ';', '"', '<', ',', '.', '>', '?', '/']

	for i in range(len(df.corpus)):
	  corp = df.corpus.iloc[i]
	  corp = nltk.wordpunct_tokenize(corp)
	  corp = [corp[i].lower() for i in range(len(corp)) if corp[i].lower() in words and corp[i].lower() not in stopwords and corp[i].lower not in spcl and len(corp[i].lower())>=3]
	  write_to_excel(corp, 'a')


# ENTER PATH TO CLEANED DATASET BELOW 
df = pd.read_csv("C:/Users/Rohan/Desktop/Lakhan Model/data.csv")
df = df.drop("Unnamed: 0", axis = 1)
df = df.dropna(axis = 0)
df = df[df["Type"] != "Invalid Document"]
target = list(df.Type)

names = list(set(target))

# USE ALL BELOW FUNCTIONS ON THE ATTACHED CLEANED DATASET TO PREP FOR TRAINING AND FURTHER CLEANING
# ONLY USE UNDERSAMPLE() FUNCTION WHILE TRAINING/PREDICTING. GET_CLEANED_DATA IS JUST A BASE FUNCTION. 
# HAD TO UNDERSAMPLE THE ABUNDANT ROWS OF DATA TO PREVENT OVERFITTING, BUT THAT MEANT REDUCING THE SIZE OF THE DATASET BY A LOT 

def get_cleaned_data():
	df_corp = pd.read_csv("C:/Users/Rohan/Desktop/Lakhan Model/cleaned.csv", sep = ',', dtype = str)
	df_list = df_corp.values.tolist()
	rows = [nltk.wordpunct_tokenize(df_list[i][0]) for i in range(len(df_list))]
	for i in range(len(rows)):
		rows[i] = [rows[i][j] for j in range(len(rows[i])) if rows[i][j]!=',']

	detokenizer = MosesDetokenizer()
	rows = [detokenizer.detokenize(rows[i], return_str = True) for i in range(len(rows))]
	del target[len(target)-1]
	del target[len(target)-1]
	return rows,target

def undersample(avg, labels, u_sample_labels):
	data, target = get_cleaned_data()

	unders = []
	unders_targets = []
	dict_l = {labels[i]:0 for i in range(len(labels))}
	for i in range(len(data)):
		dict_l[target[i]] += 1
		if target[i] in u_sample_labels and dict_l[target[i]] <= 200:
			unders.append(data[i])
			unders_targets.append(target[i])
		elif target[i] not in u_sample_labels:
			unders.append(data[i])
			unders_targets.append(target[i])

	return unders, unders_targets