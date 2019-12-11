package com.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.json4s.jackson.Serialization

object SparkUDFs extends App {

  val spark = SparkSession.builder().master("local").getOrCreate()

  val df = spark.sql("select Array(CAST(1 AS String),'rohit','null') AS data")
  val df1 = spark.sql("""select Array(Map("a","b","c","d")) AS data""")
  df.createOrReplaceTempView("df_view")
  df1.createOrReplaceTempView("df1_view")

  def uniqueKeyGenMethod(args: Seq[Any]):Long = System.nanoTime() + args.map(x => x.toString.hashCode.abs.toLong).sum

  spark.udf.register("unique_key_generator",(col:Seq[Any]) => uniqueKeyGenMethod(col))

  spark.sql("select data,unique_key_generator(data) AS unique_key from df_view").show

  val uniqueKeyGenFunction = (args:Seq[Any]) => (System.nanoTime()+args.map(_.toString.hashCode.abs.toLong).sum)
  val uniqueKeyGenerator = udf(uniqueKeyGenFunction)
  df.withColumn("unique_key",uniqueKeyGenerator(col("data"))).show

  spark.udf.register("to_json", (s: Seq[Map[String,String]]) => {
    implicit val formats = org.json4s.DefaultFormats
    Serialization.write(s)
  })

  spark.sql("select to_json(data) from df1_view").show()
}
