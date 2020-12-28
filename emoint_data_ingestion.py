'''
elasticsearch-6.7.1/bin/elasticsearch &
kibana-6.7.1-linux-x86_64/bin/kibana &
'''

import re
import csv
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from elasticsearch import Elasticsearch 

es = Elasticsearch([{'host':'0.0.0.0','port':6744}])

sc = SparkContext("local")
sqlContext = SparkSession.builder.getOrCreate()

schema = StructType()\
	.add("tweet_id",StringType(),True)\
	.add("text",StringType(),True)\
	.add("emotion",StringType(),True)\
	.add("intensity",StringType(),True)

def str2float(input):
	try:
		return float(input)
	except:
		return None

#str2float("7.567")

udf_str2float = udf(str2float, FloatType())

'''
load the files to data 
'''

data = sqlContext.read.format('csv')\
	.options(delimiter='\t')\
	.schema(schema)\
	.load('*.txt')\
	.withColumn("intensity", udf_str2float("intensity"))

'''
ingest the data to the elasticsearch index
'''
for r in data.collect():
	r1 = r.asDict()
	res = es.index(
		index = 'emoint',
		doc_type = 'doc',
		id = r["tweet_id"],
		body = r1)