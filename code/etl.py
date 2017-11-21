import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row, functions, Column
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from datetime import datetime

input1 = sys.argv[1]
input2 = sys.argv[2]
output = sys.argv[3]

def str2date(sdate):
	l = list(sdate)
	l.insert(4,'/')
	l.insert(7,'/')
	d=''.join(l)
	return datetime.strptime(d, '%Y/%m/%d')

def threshold(value,thresh=0):
	if (value > thresh):
		return 1
	else:
		return 0

spark = SparkSession.builder.appName('etl').getOrCreate()

schema = StructType([\
StructField('station',StringType(), False),\
StructField('date', StringType(), False),\
StructField('observation', StringType(), False),\
StructField('value', IntegerType(), False),\
StructField('flag1', StringType(), False),\
StructField('qflag', StringType(), False)])

schema2 = StructType([
    StructField('station', StringType(), False),
    StructField('date', DateType(), False),
    StructField('latitude', FloatType(), False),
    StructField('longitude', FloatType(), False),
    StructField('elevation', FloatType(), False),
    StructField('tmax', FloatType(), False),
])


df1=spark.read.csv(input1,schema)
df2=spark.read.csv(input2,schema2)

df1 = df1.select('*').where(df1.observation == 'PRCP').where((df1.qflag).isNull()) # filtering out other observations

df1 = df1.drop('flag1','qflag','observation') # Dropping the flags as they won't be useful for prediction, and observation since its constant

thresh = 0 # Sets the percepitation threshold

df1 = df1.withColumn('value',df1.value > thresh) #Thresholds the percepitation amount
df1 = df1.withColumn('value',(df1.value).cast(IntegerType()))

str2dateUDF = udf(str2date,DateType())
df1 = df1.withColumn('date',str2dateUDF('date')) #Converts the date from string to dateType

df_joined = df1.join(df2,['station','date']) #Joining the two tables

df_joined = df_joined.withColumn('date',(dayofyear(df1['date'])).cast(IntegerType())) #Converting date to doy

#Rearranging the columns
df_joined = df_joined.select('station','date','latitude','longitude','elevation','tmax','value')

df_joined.coalesce(1).write.csv(output, sep=' ', mode='overwrite')


#### Final Schema ####
#+-----------+----+--------+---------+---------+----+-----+
#|    station|date|latitude|longitude|elevation|tmax|value|
#+-----------+----+--------+---------+---------+----+-----+
#df_joined.printSchema()
#root
# |-- station: string (nullable = true)
# |-- date: integer (nullable = true)
# |-- latitude: float (nullable = true)
# |-- longitude: float (nullable = true)
# |-- elevation: float (nullable = true)
# |-- tmax: float (nullable = true)
# |-- value: integer (nullable = true)





