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
    StructField('latitude', DoubleType(), False),
    StructField('longitude', DoubleType(), False),
    StructField('elevation', DoubleType(), False),
    StructField('tmax', DoubleType(), False),
    StructField('precipitation', IntegerType(), False),
])

input_file = sys.argv[0]

data = spark.read.csv(input_file,sep=' ',schema=schema)

indexer = StringIndexer(inputCol = 'station', outputCol='station_id')
indexed = indexer.fit(data).transform(data)

assembler = VectorAssembler(inputCols=['station_id','dateofyear','latitude',
                                       'longitude','elevation','tmax'],
                            outputCol='features')

output = assembler.transform(indexed)

final_data = output.select('features','precipitation')

train_data,test_data = final_data.randomSplit([0.7,0.3])

layers = [4, 5, 4, 2]

trainer = MultilayerPerceptronClassifier(maxIter=100, 
                                         layers=layers, 
                                         blockSize=128, 
                                         seed=1234)

model = trainer.fit(train_data)

result = model.transform(test_data)

predictionAndLabels = result.select("prediction", "label")

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))