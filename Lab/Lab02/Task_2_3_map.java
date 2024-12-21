package com.projectgurukul.wc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.util.Pair;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;

public class Task_2_3_map extends Mapper<Object, Text, Text, FloatWritable> {
	Map<String, List<Pair<String, Float>>> items = new HashMap<String, List<Pair<String, Float>>>();
	Map<String, Float> distance = new HashMap<String, Float>();
	
	public static Map<String, List<Pair<String, Float>>> preprocessed(String value) {
    	String[] data = value.toString().split("\t");
    	String docId = data[0];
		String termId_tfidf = data[1];
		String[] items = termId_tfidf.split(","); // an item is termid - tfidf
		List<Pair<String, Float>> pairs =  new ArrayList<Pair<String, Float>>();
		
		for (String item: items) {
			String[] tmp = item.split(":");
			String termId = tmp[0];
			String tfidf = tmp[1];
			
			pairs.add(new Pair<String, Float>(termId, Float.valueOf(tfidf)));
		}

		Map<String, List<Pair<String, Float>>> res = new HashMap<String, List<Pair<String, Float>>>();
		res.put(docId, pairs);
    	return res;
    }
	
	public void setup(Context context) {
		Integer k = Integer.parseInt(context.getConfiguration().get("K"));
		// get k centroids
		for (Integer i = 0; i < 1; i++) {
			String centroid = context.getConfiguration().get("centroid" + i);
		
			Map<String, List<Pair<String, Float>>> item = preprocessed(centroid);
			for (String key: item.keySet()) {
				items.put(key, item.get(key));				
			}
			
		}
		
		
		System.out.println("\n");
	}
	
	public static Float get_value(String key, List<Pair<String, Float>> l) {
		for (Pair<String, Float> item: l) {
			if (item.getKey().equals(key)) {
				return item.getValue();
			}
		}
		return 0.0f;
	}

	
	public static void print(List<Pair<String, Float>> l) {
		for (Pair<String, Float> pair: l) {
			System.out.print(pair.getKey() + ",");
			
		}
		System.out.println("\n");
	}
	
	public static Float get_distance(List<Pair<String, Float>> v1, List<Pair<String, Float>> v2) {
		Float result = 0.0f;
		for (Pair<String, Float> item1: v1) {
			String term_id1 = item1.getKey();
			Float tf_idf1 = Float.valueOf(item1.getValue());
			
			Float tf_idf2 = get_value(term_id1, v2);
			
			result += (float)Math.pow(tf_idf1 - tf_idf2, 2);
		}
		return result;
	}
	
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		// For each point
		Map<String, List<Pair<String, Float>>> v1 = preprocessed(value.toString());
		
		String docId1;
		List<Pair<String, Float>> termId_tfidf1 = new ArrayList();
		
		for (String id: v1.keySet()) {
			docId1 = id;
			termId_tfidf1 = v1.get(docId1);
		}
		
		
		// for each centroids, get the distance of current x to all centroids in C
		for (String docId2: items.keySet()) {
			List<Pair<String, Float>> termId_tfidf2 = items.get(docId2);
			
			if (! distance.containsKey(docId2)) {
				distance.put(docId2, get_distance(termId_tfidf1, termId_tfidf2));
			} else {
				Float new_dis = get_distance(termId_tfidf1, termId_tfidf2);
				if (distance.get(docId2) < new_dis) {
					distance.put(docId2, get_distance(termId_tfidf1, termId_tfidf2));
				}
			}
			
		}
	}
	
	public void cleanup(Context context) throws IOException, InterruptedException {
		for (String id: distance.keySet()) {
			Float min_dis = distance.get(id);
			context.write(new Text(id), new FloatWritable(min_dis));
		}
	}
}
