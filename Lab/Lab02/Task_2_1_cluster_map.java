package com.projectgurukul.wc;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.util.Pair;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;

public class Task_2_1_cluster_map extends Mapper<Object, Text, Text, Text> {
			
	public void setup(Context context) {
		
	}
	
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		String[] data = value.toString().split("\t");
		String x = data[0];
		String y = data[1];
		String id = data[2];
		
		context.write(new Text(id), new Text(x + "\t" + y));
	}
	
	public void cleanup(Context context) throws IOException, InterruptedException {
		
	}
}
