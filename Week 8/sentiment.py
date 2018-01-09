
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import numpy as np 
from pyspark import SparkContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import NGram
from pyspark.ml.feature import Normalizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


sc = SparkContext("local","Simple App")
spark = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value").getOrCreate()

df = spark.read.csv('file:///home/zfar/Sentiment Analysis Dataset.csv',header=True)

df = df.select(df['ItemID'],df['SentimentText'],df['label'])

training = df.selectExpr("cast(itemID as int) id", 
                        "SentimentText", 
                        "cast(label as int) label")
                        
tokenizer = Tokenizer(inputCol="SentimentText", outputCol="words")
remover = StopWordsRemover(inputCol= tokenizer.getOutputCol() , outputCol="filtered")
ngrams = NGram(n=2, inputCol= remover.getOutputCol() , outputCol="ngrams")
hashingTF = HashingTF(inputCol=ngrams.getOutputCol(), outputCol="rawfeatures")
idf = IDF(inputCol= hashingTF.getOutputCol() , outputCol="idffeatures")
normalizer = Normalizer(inputCol= idf.getOutputCol() , outputCol="features", p=1.0)


#lr = LogisticRegression(maxIter=10, regParam=0.001)
nb = NaiveBayes(smoothing=1.0)
pipeline = Pipeline(stages=[tokenizer, remover , ngrams, hashingTF, idf , normalizer , nb])
model = pipeline.fit(training)

"""
paramGrid = ParamGridBuilder().addGrid(hashingTF.numFeatures, [10, 100, 1000]).addGrid(lr.regParam, [0.1, 0.01]).build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2) 
                          
cvModel = crossval.fit(training)
"""




test_df = spark.read.csv("file:///home/zfar/test_sentiment.csv",header=True)

test_df = test_df.selectExpr("cast(id as int) id","SentimentText","cast(label as int)label" )

test = test_df.select(test_df['id'],test_df['SentimentText'])

#prediction = cvModel.transform(test)

array_org_result = np.array(test_df.select(test_df['label']).collect())
prediction = model.transform(test)

selected = prediction.select("id", "SentimentText", "probability", "prediction")

score = []
for row in selected.collect():
	
	rid, text, prob, prediction = row

	score.append(prediction)
	#print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))

arr = np.subtract(array_org_result , score)

print("accuracy : ", np.sum(np.square(arr[0,:])/len(score)))
