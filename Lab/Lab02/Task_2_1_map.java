package com.projectgurukul.wc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.util.Pair;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;

public class Task_2_1_map extends Mapper<Object, Text, Text, IntWritable> {
	public List<Pair<Float, Float>> k_centroids = new ArrayList<Pair<Float, Float>>();
	public Boolean firstLine = true;		
	
	public void setup(Context context) {
		Integer k = Integer.parseInt(context.getConfiguration().get("K"));
		
		// get k centroids
		for (Integer i = 0; i < k; i++) {
			String centroid = context.getConfiguration().get("centroid" + i);
			String[] point = centroid.split("\t");
			k_centroids.add(new Pair<Float, Float>(Float.valueOf(point[0]), Float.valueOf(point[1])));
		}
	}
	
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		if (firstLine) {
			firstLine = false;
			return;
		}
		String[] data = value.toString().split(",");

		
		Float x = Float.valueOf(data[1]);
		Float y = Float.valueOf(data[2]);
		
		//calculate distance of 2 points
		
		Double min_distance = 0.0;
		Integer cluster_id = 0;
		Integer current_id = 0;
		Pair<Float, Float> cur_centroid = k_centroids.get(0);
		for (Pair<Float, Float> centroid: k_centroids) {
			Float c_x = centroid.getKey();
			Float c_y = centroid.getValue();
			
			Double distance = Math.sqrt((c_y - y) * (c_y - y) + (c_x - x) * (c_x - x));
			
			if (min_distance == 0.0) {
				min_distance = distance;
			} else {
				if (min_distance > distance) {
					min_distance = distance;
					cluster_id = current_id;
					cur_centroid = centroid;
				}
			}
			current_id += 1;
		}
		
		context.write(new Text(x.toString() + "\t" + y.toString()), new IntWritable(cluster_id));
	}
	
	public void cleanup(Context context) throws IOException, InterruptedException {
		
	}
}
