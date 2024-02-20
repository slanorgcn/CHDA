-- MySQL dump 10.13  Distrib 5.7.34, for osx11.0 (x86_64)
--
-- Host: ***    Database: chda
-- ------------------------------------------------------
-- Server version	5.7.29-log

--
-- Table structure for table `papercollection`
--

DROP TABLE IF EXISTS `papercollection`;
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

--
-- Table structure for table `paperreferences`
--

DROP TABLE IF EXISTS `paperreferences`;
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
