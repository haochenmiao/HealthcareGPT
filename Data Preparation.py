# Databricks notebook source
# MAGIC %md
# MAGIC ## Getting Started

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 langchain==0.1.5 databricks-vectorsearch==0.22
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query database

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW DATABASES;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN john_snow_labs_national_health_surveys.national_health_surveys;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.medicare_heart_disease_and_stroke_prevention_claims_data;

# COMMAND ----------

medicare_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.medicare_heart_disease_and_stroke_prevention_claims_data
""")

# COMMAND ----------

# Show the first few rows of the DataFrame
medicare_df.show()

# Perform some transformations or aggregations
medicare_df = medicare_df.groupBy("Disease_Topic").count()
medicare_df.show()

# COMMAND ----------

cardiovascular_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.cardiovascular_health_indicators_household_survey
""")

# Show the first few rows of the DataFrame
cardiovascular_df.show()

# Perform some transformations or aggregations
cardiovascular_df = cardiovascular_df.groupBy("Topic").count()
cardiovascular_df.show()

# COMMAND ----------

heart_disease_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.heart_disease_surveillance_system
""")

# Show the first few rows of the DataFrame
heart_disease_df.show()

# Perform some transformations or aggregations
heart_disease_df = heart_disease_df.groupBy("Topic").count()
heart_disease_df.show()

# COMMAND ----------

national_health_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.national_health_surveillance_system
""")

# Show the first few rows of the DataFrame
national_health_df.show()

# Perform some transformations or aggregations
national_health_df = national_health_df.groupBy("Topic").count()
national_health_df.show()

# COMMAND ----------

notifiable_disease1_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.notifiable_disease_surveillance_1
""")

# Show the first few rows of the DataFrame
notifiable_disease1_df.show()

# Perform some transformations or aggregations
notifiable_disease1_df = notifiable_disease1_df.groupBy("Disease").count()
notifiable_disease1_df.show()

# COMMAND ----------

notifiable_disease2_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.notifiable_disease_surveillance_2
""")

# Show the first few rows of the DataFrame
notifiable_disease2_df.show()

# Perform some transformations or aggregations
notifiable_disease2_df = notifiable_disease2_df.groupBy("Disease").count()
notifiable_disease2_df.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Start a Spark session
spark = SparkSession.builder.appName("Descriptive Column Creation").getOrCreate()

# Load the DataFrame
medicare_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.medicare_heart_disease_and_stroke_prevention_claims_data
""")

# Define the UDF for creating descriptive text
def create_full_medicare_description(year, state_abbreviation, state, disease_topic, disease_indicator, data_value_unit, 
                                     data_value, confidence_low, confidence_high, breakout_category, breakout_variables, 
                                     topic_id, indicator_id, breakout_category_id, breakout_id, location_id):
    return (f"In {year}, {state} ({state_abbreviation}), recorded a rate of {data_value} {data_value_unit} "
            f"for {disease_topic}, specifically {disease_indicator}. This rate falls within a confidence interval "
            f"from {confidence_low} to {confidence_high}. The data is categorized under {breakout_category} "
            f"with variables {breakout_variables}, associated with topic ID {topic_id} and indicator ID {indicator_id}. "
            f"Breakout categories are tagged as {breakout_category_id} with breakout ID {breakout_id}, "
            f"and the location ID for this record is {location_id}.")

# Register the UDF
full_medicare_description_udf = udf(create_full_medicare_description, StringType())

# Apply UDF to create the descriptive column
medicare_df = medicare_df.withColumn("Description", full_medicare_description_udf(
    col("Year"), col("State_Abbreviation"), col("State"), col("Disease_Topic"), col("Disease_Indicator"), 
    col("Data_Value_Unit"), col("Data_Value"), col("Confidence_Limit_Low"), col("Confidence_Limit_High"), 
    col("Breakout_Category"), col("Breakout_Variables"), col("Topic_ID"), col("Indicator_ID"), 
    col("Breakout_Category_ID"), col("Breakout_ID"), col("Location_ID")
))

# Show the transformed data to verify the descriptive column
medicare_df.show(truncate=False)
medicare_df.count()

# Remember, if you later need to perform aggregation, do it after you've used the data with descriptions.


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Start a Spark session
spark = SparkSession.builder.appName("Descriptive Column for Cardiovascular Data").getOrCreate()

# Define the UDF for creating a comprehensive description
def create_cardiovascular_description(row_id, year, location_abbreviation, location_description, data_source, 
                                      priority_area_1, priority_area_2, priority_area_3, category, topic, indicator, 
                                      data_value_type, data_value_unit, data_value, data_value_alt, 
                                      data_value_footnote_symbol, data_value_footnote, confidence_limit_low, 
                                      confidence_limit_high, break_out_category, break_out, category_id, topic_id, 
                                      indicator_id, data_value_type_id, break_out_category_id, break_out_id, location_id):
    return (f"Record ID {row_id} from {year} in {location_description} ({location_abbreviation}), sourced from {data_source}, "
            f"falls under {category} with a focus on {topic}, specifically the indicator {indicator}. "
            f"This {data_value_type} data shows a value of {data_value} {data_value_unit} (alternate value {data_value_alt}), "
            f"noted under {break_out_category} '{break_out}'. "
            f"Data confidence intervals range from {confidence_limit_low} to {confidence_limit_high}. "
            f"Priority areas include {priority_area_1}, {priority_area_2}, and {priority_area_3}. "
            f"Footnote symbol: {data_value_footnote_symbol}, Footnote detail: {data_value_footnote}. "
            f"Data categories and IDs: Category ID {category_id}, Topic ID {topic_id}, Indicator ID {indicator_id}, "
            f"Data Value Type ID {data_value_type_id}, Break Out Category ID {break_out_category_id}, Break Out ID {break_out_id}. "
            f"Location ID for this data is {location_id}.")

# Register the UDF
full_cardio_description_udf = udf(create_cardiovascular_description, StringType())

# Load the DataFrame
cardiovascular_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.cardiovascular_health_indicators_household_survey
""")

# Apply UDF to create the descriptive column
cardiovascular_df = cardiovascular_df.withColumn("Description", full_cardio_description_udf(
    col("Row_Id"), col("Year"), col("Location_Abbreviation"), col("Location_Description"), col("Data_Source"), 
    col("Priority_Area_1"), col("Priority_Area_2"), col("Priority_Area_3"), col("Category"), col("Topic"), col("Indicator"), 
    col("Data_Value_Type"), col("Data_Value_Unit"), col("Data_Value"), col("Data_Value_Alt"), col("Data_Value_Footnote_Symbol"), 
    col("Data_Value_Footnote"), col("Confidence_Limit_Low"), col("Confidence_Limit_High"), col("Break_Out_Category"), 
    col("Break_Out"), col("Category_ID"), col("Topic_ID"), col("Indicator_ID"), col("Data_Value_Type_ID"), 
    col("Break_Out_Category_ID"), col("Break_Out_ID"), col("Location_ID")
))

# Show the transformed data to verify the descriptive column
cardiovascular_df.show(truncate=False)
cardiovascular_df.count()

# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Start a Spark session
spark = SparkSession.builder.appName("Descriptive Column for heart disease Data").getOrCreate()

# Define the UDF for creating a comprehensive description for heart disease data
def create_heart_disease_description(row_id, year, location_abbreviation, location_description, data_source, 
                                     priority_area_1, priority_area_2, priority_area_3, category, topic, indicator, 
                                     data_value_type, data_value, data_value_alt, data_value_footnote_symbol, 
                                     data_value_footnote, confidence_limit_low, confidence_limit_high, 
                                     break_out_category, break_out, category_id, topic_id, indicator_id, 
                                     data_value_type_id, breakout_category_id, break_out_id, location_id, latitude, longitude):
    return (f"Record ID {row_id} from {year} in {location_description} ({location_abbreviation}), sourced from {data_source}, "
            f"falls under {category} with a focus on {topic} - {indicator}. Data type: {data_value_type}, value: {data_value} "
            f"{data_value_alt} with footnotes {data_value_footnote_symbol} {data_value_footnote}. Confidence intervals range "
            f"from {confidence_limit_low} to {confidence_limit_high}. Data is categorized under {break_out_category} as {break_out}. "
            f"Priority areas: {priority_area_1}, {priority_area_2}, {priority_area_3}. Category ID {category_id}, Topic ID {topic_id}, "
            f"Indicator ID {indicator_id}, Data Value Type ID {data_value_type_id}, Break Out Category ID {breakout_category_id}, "
            f"Break Out ID {break_out_id}. Location ID {location_id}, coordinates {latitude}, {longitude}.")

# Register the UDF
heart_disease_description_udf = udf(create_heart_disease_description, StringType())

heart_disease_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.heart_disease_surveillance_system
""")

heart_disease_df = heart_disease_df.withColumn("Description", heart_disease_description_udf(
    col("Row_Id"), col("Year"), col("Location_Abbreviation"), col("Location_Description"), col("Data_Source"),
    col("Priority_Area_1"), col("Priority_Area_2"), col("Priority_Area_3"), col("Category"), col("Topic"), col("Indicator"),
    col("Data_Value_Type"), col("Data_Value"), col("Data_Value_Alt"), col("Data_Value_Footnote_Symbol"),
    col("Data_Value_Footnote"), col("Confidence_Limit_Low"), col("Confidence_Limit_High"), col("Break_Out_Category"),
    col("Break_Out"), col("Category_Id"), col("Topic_ID"), col("Indicator_ID"), col("Data_Value_Type_ID"),
    col("Breakout_Category_ID"), col("Break_Out_ID"), col("Location_ID"), col("Latitude"), col("Longitude")
))

# Show the transformed data to verify the descriptive column
heart_disease_df.show(truncate=False)
heart_disease_df.count()


# COMMAND ----------

# Start a Spark session
spark = SparkSession.builder.appName("Descriptive Column for national health Data").getOrCreate()

# Define the UDF for creating a comprehensive description for national health data
def create_national_health_description(nhanes_row_id, year, location_abbreviation, location_description, data_source,
                                       priority_area_1, priority_area_2, priority_area_3, priority_area_4, category, topic, indicator,
                                       data_value_type, data_value_unit, data_value, data_value_alt, data_value_footnote_symbol,
                                       data_value_footnote, confidence_interval_low, confidence_interval_high, break_out_category,
                                       break_out, category_id, topic_id, indicator_id, data_value_type_id, break_out_category_id,
                                       break_out_id, location_id):
    return (f"Record {nhanes_row_id} from {year} in {location_description} ({location_abbreviation}), sourced from {data_source}, "
            f"is categorized under {category} focusing on {topic} with the specific indicator {indicator}. "
            f"This data is recorded as {data_value_type} ({data_value_unit}) with a value of {data_value} (alternate value: {data_value_alt}), "
            f"with footnotes {data_value_footnote_symbol} {data_value_footnote}. Confidence intervals range from "
            f"{confidence_interval_low} to {confidence_interval_high}. Data falls under the break out category {break_out_category} as {break_out}. "
            f"Priority areas are {priority_area_1}, {priority_area_2}, {priority_area_3}, and {priority_area_4}. "
            f"Category ID: {category_id}, Topic ID: {topic_id}, Indicator ID: {indicator_id}, "
            f"Data Value Type ID: {data_value_type_id}, Break Out Category ID: {break_out_category_id}, Break Out ID: {break_out_id}, "
            f"Location ID: {location_id}.")

# Register the UDF
national_health_description_udf = udf(create_national_health_description, StringType())

national_health_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.national_health_surveillance_system
""")


national_health_df = national_health_df.withColumn("Description", national_health_description_udf(
    col("NHANES_Row_Id"), col("Year"), col("Location_Abbreviation"), col("Location_Description"), col("Data_Source"),
    col("Priority_Area_1"), col("Priority_Area_2"), col("Priority_Area_3"), col("Priority_Area_4"), col("Category"), 
    col("Topic"), col("Indicator"), col("Data_Value_Type"), col("Data_Value_Unit"), col("Data_Value"), 
    col("Data_Value_Alt"), col("Data_Value_Footnote_Symbol"), col("Data_Value_Footnote"), col("Confidence_Interval_Low"), 
    col("Confidence_Interval_High"), col("Break_Out_Category"), col("Break_Out"), col("Category_ID"), col("Topic_ID"), 
    col("Indicator_ID"), col("Data_Value_Type_ID"), col("Break_Out_Category_ID"), col("Break_Out_ID"), col("Location_ID")
))

# Show the transformed data to verify the descriptive column
national_health_df.show(truncate=False)
national_health_df.count()

# COMMAND ----------

# Start a Spark session
spark = SparkSession.builder.appName("Descriptive Column for notifiable disease1 Data").getOrCreate()

# Define the UDF for creating a comprehensive description
def create_notifiable_disease_description(epi_week, state_abbreviation, location, location_type, disease, cases, incidence):
    return (f"During epidemiological week {epi_week}, {location} ({state_abbreviation}), a {location_type}, "
            f"reported {cases} cases of {disease}, with an incidence rate of {incidence} per 100,000 people.")

# Register the UDF
notifiable_disease_description_udf = udf(create_notifiable_disease_description, StringType())

notifiable_disease1_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.notifiable_disease_surveillance_1
""")

notifiable_disease1_df = notifiable_disease1_df.withColumn("Description", notifiable_disease_description_udf(
    col("Epi_Week"), col("State_Abbreviation"), col("Location"), col("Location_Type"),
    col("Disease"), col("Cases"), col("Incidence")
))

# Show the transformed data to verify the descriptive column
notifiable_disease1_df.show(truncate=False)
notifiable_disease1_df.count()

# COMMAND ----------

# Start a Spark session
spark = SparkSession.builder.appName("Descriptive Column for notifiable disease2 Data").getOrCreate()

# Define the UDF for creating a comprehensive description for notifiable disease data
def create_notifiable_disease2_description(epi_week, state_abbreviation, location, location_type, disease, event, cases, from_date, to_date, url):
    return (f"In epidemiological week {epi_week}, {location} ({state_abbreviation}), a {location_type}, "
            f"reported {cases} cases of {disease} due to {event}. The cases were reported from {from_date} to {to_date}. "
            f"More details can be found at {url}.")

# Register the UDF
notifiable_disease2_description_udf = udf(create_notifiable_disease2_description, StringType())

notifiable_disease2_df = spark.sql("""
SELECT * FROM john_snow_labs_national_health_surveys.national_health_surveys.notifiable_disease_surveillance_2
""")

notifiable_disease2_df = notifiable_disease2_df.withColumn("Description", notifiable_disease2_description_udf(
    col("Epi_Week"), col("State_Abbreviation"), col("Location"), col("Location_Type"),
    col("Disease"), col("Event"), col("Cases"), col("From_Date"), col("To_Date"), col("Url")
))

# Show the transformed data to verify the descriptive column
notifiable_disease2_df.show(truncate=False)
notifiable_disease2_df.count()

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame

medicare_desc = medicare_df.select("Description")
cardiovascular_desc = cardiovascular_df.select("Description")
heart_disease_desc = heart_disease_df.select("Description")
national_health_desc = national_health_df.select("Description")
notifiable_disease1_desc = notifiable_disease1_df.select("Description")
notifiable_disease2_desc = notifiable_disease2_df.select("Description")

# Function to union multiple DataFrames
def unionAll(*dfs):
    return reduce(DataFrame.union, dfs)

# Combine all description DataFrames into one
combined_descriptions_df = unionAll(medicare_desc, cardiovascular_desc, heart_disease_desc, 
                                    national_health_desc, notifiable_disease1_desc, notifiable_disease2_desc)

combined_descriptions_df.show(truncate=False)
combined_descriptions_df.count()

# COMMAND ----------


