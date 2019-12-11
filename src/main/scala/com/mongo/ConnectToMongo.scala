package com.mongo

import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.sql.SparkSession

object ConnectToMongo {
  private def registerCollection(sparkSession: SparkSession,collection:String): Unit = {
    val readConfig = ReadConfig(Map("uri" -> "mongodb://user:pass@localhost:27017/db?authSource=db&authMechanism=SCRAM-SHA-1"))
    val collection = MongoSpark.builder()
      .sparkSession(sparkSession).readConfig(readConfig.copy(databaseName = "db", collectionName = "test_topic")).build().toDF()
    collection.createOrReplaceTempView("test_collection")
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").getOrCreate()
    registerCollection(spark,"test_collection")
    spark.sql("""select * from test_collection limit 1""").show()
  }

}
