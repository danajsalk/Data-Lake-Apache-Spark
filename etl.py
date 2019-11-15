import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, \
    date_format

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    loads song data from s3 and extracts song table data and artist table data

    params
    spark: spark instance
    input_data (json): path to input data
    output_data (parquet): location in s3
    """
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'

    # read song data file and create view for queries
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df['song_id', 'title', 'artist_id', 'year', 'duration']
    songs_table = songs_table.dropDuplicates(['song_id'])

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode('overwrite').partitionBy("year",
                                                    "artist_id").parquet(
        output_data + 'songs_table/')

    # extract columns to create artists table
    artists_table = df[
        'artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude']
    artists_table = artists_table.dropDuplicates(['artist_id'])

    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(
        output_data + 'artists_table/')


def process_log_data(spark, input_data, output_data):
    """
    loads song data from s3 and extracts user table data and log table data

    params
    spark: spark instance
    input_data (json): path to input data
    output_data (parquet): location in s3
    """
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    songplays_table = df[
        'ts', 'userId', 'level', 'sessionId', 'location', 'userAgent']

    # extract columns for users table    
    users_table = df['userId', 'firstName', 'lastName', 'gender', 'level']
    users_table = users_table.dropDuplicates(['userId'])

    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data + 'users_table/')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x) / 1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000.0)))
    df = df.withColumn("datetime", get_datetime(df.ts))

    # extract columns to create time table
    time_table = df.select(
        col('datetime').alias('start_time'),
        hour('datetime').alias('hour'),
        dayofmonth('datetime').alias('day'),
        weekofyear('datetime').alias('week'),
        month('datetime').alias('month'),
        year('datetime').alias('year')
    )
    time_table = time_table.dropDuplicates(['start_time'])

    # write time table to parquet files partitioned by year and month
    time_table.write.mode('overwrite').partitionBy("year", "month").parquet(
        output_data + 'time_table')

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + 'songs_table/')

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql("""
    SELECT to_timestamp(l.ts/1000) 'start_time'
        , month(to_timestamp(l.ts/1000)) 'month'
        , year(to_timestamp(l.ts/1000)) 'year'
        , l.userId 'user_id'
        , l.level 'level'
        , s.song_id
        , s.artist_id
        , l.sessionId 'session_id'
        , l.location 
        , l.userAgent 'user_agent'
    FROM log_table_view l
        JOIN song_table_view s ON l.artist = s.artist_name 
            AND l.song = s.title
    """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode('overwrite').partitionBy("year",
                                                        "month").parquet(
        output_data + 'songplays_table/')


def main():
    """
    Process S3 songs and events data, transform it into demensional tables, 
    load back to S3
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-dend/dloutput/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
