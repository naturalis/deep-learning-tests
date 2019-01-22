# Martin collectie
## 1) DwCA download 
**query**   
{
  "conditions" : [
    { "field" : "theme", "operator" : "=", "value" : "martin" },
    { "field" : "associatedMultiMediaUris.format", "operator" : "!="  }
  ],
  "logicalOperator" : "AND",
  "fields" : [ "gatheringEvent.country", "identifications.scientificName", "associatedMultiMediaUris"  ],
  "from" : 0,
  "size" : 100000
}

**full URL**  
[http://api.biodiversitydata.nl/v2/specimen/dwca/query/?_querySpec= ... ](http://api.biodiversitydata.nl/v2/specimen/dwca/query/?_querySpec=%7B%20%20%20%22conditions%22%20%3A%20%5B%0A%20%20%20%20%7B%20%22field%22%20%3A%20%22theme%22%2C%20%22operator%22%20%3A%20%22%3D%22%2C%20%22value%22%20%3A%20%22martin%22%20%7D%2C%0A%20%20%20%20%7B%20%22field%22%20%3A%20%22associatedMultiMediaUris.format%22%2C%20%22operator%22%20%3A%20%22%21%3D%22%20%20%7D%0A%20%20%5D%2C%0A%20%20%22logicalOperator%22%20%3A%20%22AND%22%2C%0A%20%20%22fields%22%20%3A%20%5B%20%22gatheringEvent.country%22%2C%20%22identifications.scientificName%22%2C%20%22associatedMultiMediaUris%22%20%20%5D%2C%0A%20%20%22from%22%20%3A%200%2C%0A%20%20%22size%22%20%3A%20100000%0A%7D)


**files**  
download and unzip:
* eml.xml
* meta.xml
* Occurrence.csv

opened Occurrence.csv in spreadsheet
took out column of names
made them distinct


