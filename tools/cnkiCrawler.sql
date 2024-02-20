-- MySQL dump 10.13  Distrib 5.7.34, for osx11.0 (x86_64)
--
-- Host: ***    Database: chda
-- ------------------------------------------------------
-- Server version	5.7.29-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `papercollection`
--

DROP TABLE IF EXISTS `papercollection`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `papercollection` (
  `uuid` varchar(255) NOT NULL,
  `paper_title` varchar(255) DEFAULT NULL,
  `publication_year` varchar(255) DEFAULT NULL,
  `journal_name` varchar(255) DEFAULT NULL,
  `authors` text,
  `abstract` text,
  `created_timestamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `status` tinyint(4) DEFAULT '1',
  `is_deleted` tinyint(4) DEFAULT '0',
  PRIMARY KEY (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `paperreferences`
--

DROP TABLE IF EXISTS `paperreferences`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `paperreferences` (
  `paper_uuid` varchar(255) NOT NULL,
  `referenced_paper_uuid` varchar(255) NOT NULL,
  `created_timestamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `status` tinyint(4) DEFAULT '1',
  `is_deleted` tinyint(4) DEFAULT '0',
  PRIMARY KEY (`paper_uuid`,`referenced_paper_uuid`),
  KEY `referenced_paper_uuid` (`referenced_paper_uuid`),
  CONSTRAINT `paperreferences_ibfk_1` FOREIGN KEY (`paper_uuid`) REFERENCES `papercollection` (`uuid`),
  CONSTRAINT `paperreferences_ibfk_2` FOREIGN KEY (`referenced_paper_uuid`) REFERENCES `papercollection` (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-02-20 15:08:28
