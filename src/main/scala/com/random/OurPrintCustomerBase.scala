package com.random

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

object OurPrintCustomerBase extends App {
  val spark = SparkSession.builder().master("local[*]").getOrCreate()

    val second_year = spark.read.option("header","true").csv("/Users/rnimmala/Desktop/dataset/2_YR.csv").withColumn("semester",lit("2-2"))
    val third_year = spark.read.option("header","true").csv("/Users/rnimmala/Desktop/dataset/3_YR.csv").withColumn("semester",lit("3-1"))
    val fourth_year = spark.read.option("header","true").csv("/Users/rnimmala/Desktop/dataset/4_YR.csv").withColumn("semester",lit("3-2"))

    val all_data = second_year.union(third_year).union(fourth_year).persist(StorageLevel.DISK_ONLY)

    all_data.createOrReplaceTempView("all_data")

  val query = spark.sql(
    """
      |select
      |  name,
      |    CASE
      |    WHEN (size(name_array) >= 3 AND mod==1) THEN CONCAT(lower(name_array[1]),'_',lower(name_array[0]),lower(name_array[2]),'@outlook.com')
      |    WHEN (size(name_array) < 3 AND mod==1) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |    WHEN (size(name_array) == 1 AND mod==1) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==2) THEN CONCAT(lower(name_array[2]),'_',lower(name_array[1]),'_',lower(name_array[0]),age,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==2) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'.',roll_no,'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==2) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==3) THEN CONCAT(lower(name_array[0]),'.',lower(name_array[2]),ceil(rand()*3+268),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==3) THEN CONCAT(lower(name_array[1]),'_',lower(name_array[0]),lower(name_array[1]),mod,ceil(rand()*3+1998),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==3) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==4) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),roll_no,ceil(rand()*89+19),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==4) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),age,'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==4) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==5) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==5) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,ceil(rand()*84+122),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==5) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==6) THEN CONCAT(lower(name_array[2]),'_',lower(name_array[0]),'_',lower(name_array[2]),age,'@yahoo.co.in')
      |    WHEN (size(name_array) < 3 AND mod==6) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),ceil(rand()*3+1998),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==6) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==7) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),ceil(rand()*84+122),'@outlook.com')
      |    WHEN (size(name_array) < 3 AND mod==7) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,ceil(rand()*84+1322),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==7) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==8) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'.',lower(name_array[1]),'@outlook.com')
      |    WHEN (size(name_array) < 3 AND mod==8) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,ceil(rand()*8+322),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==8) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==9) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),age,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==9) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),ceil(rand()*8+352),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==9) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==10) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==10) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==10) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==11) THEN CONCAT(lower(name_array[0]),'.',lower(name_array[2]),ceil(rand()*3+1998),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==11) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),ceil(rand()*3+1998),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==11) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,ceil(rand()*3+19),age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==12) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),'_',lower(course),ceil(rand()*8+352),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==12) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==12) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],2,4)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==13) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),age,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==13) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'.',lower(course),mod,ceil(rand()*8+352),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==13) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==14) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),roll_no,'@yahoo.co.in')
      |    WHEN (size(name_array) < 3 AND mod==14) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),lower(course),ceil(rand()*3+268),'@yahoo.in')
      |    WHEN (size(name_array) == 1 AND mod==14) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==15) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),mod,ceil(rand()*8+352),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==15) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==15) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==16) THEN CONCAT(lower(name_array[0]),'.',lower(name_array[2]),ceil(rand()*3+1998),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==16) THEN CONCAT(lower(name_array[1]),'_',lower(name_array[0]),age,lower(course),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==16) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==17) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),age,'@outlook.com')
      |    WHEN (size(name_array) < 3 AND mod==17) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),lower(name_array[1]),mod,ceil(rand()*3+18),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==17) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==18) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),roll_no,mod,ceil(rand()*8+352),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==18) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,'@yahoo.in')
      |    WHEN (size(name_array) == 1 AND mod==18) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==19) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==19) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),age,mod,ceil(rand()*8+352),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==19) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==20) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),age,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==20) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==20) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==21) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==21) THEN CONCAT(lower(name_array[1]),'_',lower(name_array[0]),'.',lower(course),roll_no,'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==21) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==22) THEN CONCAT(lower(name_array[0]),age,'_',lower(name_array[1]),'_',lower(name_array[2]),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==22) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),ceil(rand()*3+18),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==22) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==23) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==23) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==23) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==24) THEN CONCAT(lower(name_array[0]),'.',lower(name_array[2]),ceil(rand()*3+1998),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==24) THEN CONCAT(lower(name_array[1]),'_',lower(name_array[0]),ceil(rand()*3+378),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==24) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==25) THEN CONCAT(lower(name_array[0]),age,'_',lower(name_array[1]),'_',lower(name_array[2]),age,mod,ceil(rand()*8+352),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==25) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),roll_no,'@yahoo.in')
      |    WHEN (size(name_array) == 1 AND mod==25) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==26) THEN CONCAT(lower(name_array[0]),'_',lower(name_array[1]),'_',lower(name_array[2]),lower(course),roll_no,'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==26) THEN CONCAT(lower(name_array[1]),'_',lower(name_array[0]),age,mod,ceil(rand()*8+352),'@gmail.com')
      |    WHEN (size(name_array) == 1 AND mod==26) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |
      |    WHEN (size(name_array) >= 3 AND mod==27) THEN CONCAT(lower(name_array[0]),'.',lower(name_array[1]),lower(name_array[2]),ceil(rand()*3+908),'@gmail.com')
      |    WHEN (size(name_array) < 3 AND mod==27) THEN CONCAT(lower(name_array[1]),'_',lower(name_array[0]),ceil(rand()*3+908),age,roll_no,'@yahoo.com')
      |    WHEN (size(name_array) == 1 AND mod==27) THEN CONCAT(lower(name_array[0]),'_',lower(substr(name_array[0],1,3)),roll_no,age,'@outlook.com')
      |    ELSE CONCAT(lower(name_array[0]),'.',lower(course),age,ceil(rand()*3+20),lower(name_array[0]),age,roll_no,'@yahoo.com')
      |  END AS email_id,
      |  mobile_number,
      |  gender,
      |  age,
      |  college,
      |  course,
      |  semester
      |  -- sno,
      |  -- mod,
      |  -- hno,
      |  -- roll_no,
      |  -- concat_ws(',',name_array),
      |  -- size(name_array),
      |  -- reg,
      |FROM
      |(select
      |    CASE
      |      WHEN (sno%107 == 0) THEN 1
      |      WHEN (sno%103 == 0) THEN 2
      |      WHEN (sno%101 == 0) THEN 3
      |      WHEN (sno%97 == 0) THEN 4
      |      WHEN (sno%89 == 0) THEN 5
      |      WHEN (sno%83 == 0) THEN 6
      |      WHEN (sno%79 == 0) THEN 7
      |      WHEN (sno%73 == 0) THEN 8
      |      WHEN (sno%71 == 0) THEN 9
      |      WHEN (sno%61 == 0) THEN 10
      |      WHEN (sno%59 == 0) THEN 11
      |      WHEN (sno%53 == 0) THEN 12
      |      WHEN (sno%47 == 0) THEN 13
      |      WHEN (sno%43 == 0) THEN 14
      |      WHEN (sno%41 == 0) THEN 15
      |      WHEN (sno%37 == 0) THEN 16
      |      WHEN (sno%31 == 0) THEN 17
      |      WHEN (sno%29 == 0) THEN 18
      |      WHEN (sno%23 == 0) THEN 19
      |      WHEN (sno%19 == 0) THEN 20
      |      WHEN (sno%17 == 0) THEN 21
      |      WHEN (sno%13 == 0) THEN 22
      |      WHEN (sno%11 == 0) THEN 23
      |      WHEN (sno%7 == 0) THEN 24
      |      WHEN (sno%5 == 0) THEN 25
      |      WHEN (sno%3 == 0) THEN 26
      |      WHEN (sno%2 == 0) THEN 27
      |    ELSE 1
      |    END AS mod,
      |    reg,
      |    hno,
      |    sno,
      |    name,
      |    SPLIT(name,' ') AS name_array,
      |    SUBSTR(hno,8) AS roll_no,
      |    CONCAT(ceil(rand()*4+5), ceil(rand()*8+1),ceil(rand()*7+1),ceil(rand()*6+1),ceil(rand()*8+1),ceil(rand()*8+1),ceil(rand()*5+1),ceil(rand()*8+1),ceil(rand()*4+1),ceil(rand()*3+1)) AS mobile_number,
      |    '' as gender,
      |    CONCAT(ceil(rand()*3+18)) AS age,
      |    'GNIT' AS college,
      |    split(branch,'-')[0] AS course,
      |    semester
      |from all_data)
    """.stripMargin)

  query.withColumn("source",lit("Y")).coalesce(1).write.option("header","true").csv("/Users/rnimmala/Desktop/dataset/processed_data")
}
