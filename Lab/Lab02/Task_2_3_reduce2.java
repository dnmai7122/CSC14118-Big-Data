package com.projectgurukul.wc;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;

public class Task_2_3_reduce2 extends Reducer<Text, FloatWritable, Text, FloatWritable> {
	TreeMap<Float, String> tm = new TreeMap<Float, String>(Collections.reverseOrder());
	Integer l = 0;
	public void setup(Context context) {
		l = context.getConfiguration().getInt("l", 0);
	}
	
	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
		
		// each point has only 1 probability
		for (Text value: values) {
			Float p = Float.valueOf(value.toString());
			tm.put(p, key.toString());
		}
	}
	
	public void cleanup(Context context) throws IOException, InterruptedException {
		Integer count = 0;
		for (Float dis: tm.keySet()) {
			String docId = tm.get(dis);
			context.write(new Text(docId), new FloatWritable(dis));
			count += 1;
			if (count == l) {
				break;
			}
		}
	}
}
