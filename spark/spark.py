from pyspark.sql import SparkSession
from pyspark import SparkConf
from distkeras.utils import *

application_name = "Scatter Experiments"
local = True

if local:
    # Tell master to use local resources.
    master = "local[*]"
    num_processes = 3
    num_executors = 1
else:
    # Tell master to use YARN.
    master = "yarn-client"
    num_executors = 20
    num_processes = 1

conf = SparkConf()
conf.set("spark.app.name", application_name)
conf.set("spark.master", master)
conf.set("spark.executor.cores", repr(num_processes))
conf.set("spark.executor.instances", repr(num_executors))
conf.set("spark.executor.memory", "4g")
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.local.dir", "/tmp/" + get_os_username() + "/dist-keras")

sc = SparkSession.builder.config(conf=conf) \
    .appName(application_name) \
    .getOrCreate()