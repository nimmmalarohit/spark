package com.kafka

import java.text.SimpleDateFormat
import java.util.{Date, Properties, TimeZone}

import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}


object SendKafkaMessage {

  def main(args: Array[String]): Unit = {
    sendKafkaNotificationMessage(List("writeback_streaming_test"))
  }

  private def sendKafkaNotificationMessage(topicList: List[String]): Unit = {
    val formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    formatter.setTimeZone(TimeZone.getTimeZone("GMT"))

    val message: String = s"""{"message":{ "senderHost": "localhost", "finishedTime": "2019-02-05 04:00:08" }}"""
    println(message)

    for (topic <- topicList) {
      System.out.println(" ======= " + topic + " ============ ")
      val properties = new Properties()
      properties.put("bootstrap.servers", "localhost:9092")
      properties.put("serializer.class", "kafka.serializer.StringEncoder")
      properties.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
      properties.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

      val producer = new KafkaProducer[String,String](properties)

      producer.send(new ProducerRecord[String,String](topic,message))
      producer.close()




//    val producerConfig = new ProducerConfig(properties)
//      val producer = new kafka.javaapi.producer.Producer[String, String](producerConfig)
//      val messageToSend = new KeyedMessage[String, String](topic, message)
//      producer.send(messageToSend)
//      producer.close
//      System.out.println("sent  ==> " + message)
    }
  }

}
