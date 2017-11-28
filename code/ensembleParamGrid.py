import pandas as pd
import numpy as np
import sys

from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


spark = SparkSession.builder.appName('ensemble_with_param_grid').getOrCreate()

# For a row with columns as predictions, choose the class based on the majority
def committee_voting(dataframe_row):
    total_values = dataframe_row.values.sum()
    if total_values >= (num_of_experts / 2):
        return 1
    else:
        return 0 

schema = StructType([
    StructField('station', StringType(), False),
    StructField('dateofyear', FloatType(), False),
    StructField('latitude', FloatType(), False),
    StructField('longitude', FloatType(), False),
    StructField('elevation', FloatType(), False),
    StructField('tmax', FloatType(), False),
    StructField('label', IntegerType(), False),
])

input_file = sys.argv[1] 

data = spark.read.csv(input_file,sep=' ',schema=schema)

#drop the rows with 'nan' values (if exist)
data = data.na.drop()

# create a column with the composition of all the features
assembler = VectorAssembler(inputCols=['dateofyear','latitude', 
                                       'longitude','elevation',
                                       'tmax'],
                            outputCol='features')
output = assembler.transform(data)

# create spark df with the columns features and label
final_data = output.select('features','label')

##### GRID PARAMETER BUILDER

mlpc = MultilayerPerceptronClassifier(blockSize=128, seed=1234)

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
# paramGrid = ParamGridBuilder() \
# 	.addGrid(mlpc.maxIter, [5, 1000,1000,2000]) \
#     .addGrid(mlpc.layers, [[2,2,2],[2,5,2],[3,6,3,2]])\
#     .addGrid(mlpc.stepSize, [0.5,0.2,0.1,0.05,0.02])\
#     .addGrid(mlpc.solver, ['l-bfgs', 'gd'])\
#     .addGrid(mlpc.tol, [1e-06, 1e-05, 1e-04])\
#     .build()

# SIMPLER COMBINATION FOR TEST
paramGrid = ParamGridBuilder() \
    .addGrid(mlpc.maxIter, [5, 10]) \
    .addGrid(mlpc.layers, [[5,2,2],[5,5,2]]) \
    .addGrid(mlpc.stepSize, [0.5,0.2]) \
    .build()



# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
tvs = TrainValidationSplit(estimator=mlpc,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
model = tvs.fit(train_data)

# Save the parameters of the best model into variables
bestmodel = model.bestModel
layers = list(bestmodel._java_obj.parent().getLayers())
iters = bestmodel._java_obj.parent().getMaxIter()
solver = bestmodel._java_obj.parent().getSolver()
tol = bestmodel._java_obj.parent().getTol()
lr = bestmodel._java_obj.parent().getStepSize()

# Define number of experts (neural nets) to be trained
num_of_experts = 10

# Dictionary to store the models trained for each expert
dict_of_models = dict()

# List of dataframes for each of the experts
dataframes = final_data.randomSplit([1.0 for x in range(num_of_experts)],seed=1234)

# Get the models for each expert using the parameters of the best model defined above
for expert in range(num_of_experts):

    train_data,test_data = dataframes[expert].randomSplit([0.8,0.2])
    
    trainer = MultilayerPerceptronClassifier(maxIter=iters, 
                                             layers=layers,
                                             stepSize=lr,
                                             blockSize=128,
                                             solver=solver, ##### include in the other version
                                             tol=tol, #### include in the other version
                                             seed=1234)
    model = trainer.fit(train_data)
    dict_of_models[expert] = model

# Dictionary to store the predictions of the full dataset for each trained expert
dict_of_predictions = dict()

# Iterate through the expert and predict the values of each dataset
for expert in range(num_of_experts):
    dict_of_predictions[expert] = dict_of_models[expert].transform(final_data)

# Create a pandas dataframe whose columns are each predictions of each expert
evaluations = pd.concat([dict_of_predictions[x].toPandas().prediction for x in range(num_of_experts)],axis=1)

# Rename the prediction columns to reference the number of the expert (sequential)
evaluations.columns = ['prediction'+str(x) for x in range(num_of_experts)]

# Create a list with the result of the committee votation
evaluations_vote = []
for index, row in evaluations.iterrows():
    evaluations_vote.append(committee_voting(row))


# Create a dataframe with the label and the prediction
predictionAndLabels = pd.concat([final_data.toPandas().label, pd.DataFrame(evaluations_vote)],axis=1)
predictionAndLabels.columns = ['label','predictions']

# Create tha colum 'evaluation' to check if the prediction is correct
predictionAndLabels['evaluation'] = np.where(predictionAndLabels.label == predictionAndLabels.predictions, 1, 0)

accuracy = predictionAndLabels.evaluation.sum() / predictionAndLabels.shape[0]

print("total os 0's: " + str(predictionAndLabels.shape[0] - predictionAndLabels.evaluation.sum()) )
print("total os 1's: " + str(predictionAndLabels.evaluation.sum()) )
print('accuracy: ' + str(accuracy))

