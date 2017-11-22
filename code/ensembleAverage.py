from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StringIndexer
import sys

spark = SparkSession.builder.appName('ensemble average').getOrCreate()

schema = StructType([
    StructField('station', StringType(), False),
    StructField('dateofyear', IntegerType(), False),
    StructField('latitude', FloatType(), False),
    StructField('longitude', FloatType(), False),
    StructField('elevation', FloatType(), False),
    StructField('tmax', FloatType(), False),
    StructField('label', IntegerType(), False),
])

input_file = sys.argv[1]

data = spark.read.csv(input_file,sep=' ',schema=schema)

data = data.na.drop()

assembler = VectorAssembler(inputCols=['dateofyear','latitude', 
                                       'longitude','elevation',
                                       'tmax'],
                            outputCol='features')

output = assembler.transform(data)

final_data = output.select('features','label')

train_data,test_data = final_data.randomSplit([0.7,0.3])

layers = [5, 3, 3, 2]

trainer = MultilayerPerceptronClassifier(maxIter=100, 
                                         layers=layers, 
                                         blockSize=128, 
                                         seed=1234)

model = trainer.fit(train_data)

result = model.transform(test_data)

predictionAndLabels = result.select("prediction", "label")

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
