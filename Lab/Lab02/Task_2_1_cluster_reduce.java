package com.projectgurukul.wc;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.io.FloatWritable;

public class Task_2_1_cluster_reduce extends Reducer<Text, Text, Text, FloatWritable> {
	
	public void setup(Context context) {
		
	}
	
	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
		Float x_mean = 0.0f;
		Float y_mean = 0.0f;
		Integer total = 0;
		for (Text value: values) {
			String[] data = value.toString().split("\t");
			Float x = Float.valueOf(data[0]);
			Float y = Float.valueOf(data[1]);
			
			x_mean += x;
			y_mean += y;
			total += 1;
		}
		
		x_mean /= total;
		y_mean /= total;
		
		context.write(new Text(x_mean.toString()), new FloatWritable(y_mean));
	}
	
	public void cleanup(Context context) throws IOException, InterruptedException {
		
	}
}
