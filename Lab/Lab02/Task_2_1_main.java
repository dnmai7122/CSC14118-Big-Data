package com.projectgurukul.wc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.util.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Task_2_1_main {
	public static void main(String[] args) throws Exception {
		Integer K = 3;
		Integer INTERATIONS = 20;
		Configuration conf = new Configuration();
		Path input_path = new Path(args[0]);
//		Path output_path = new Path(args[1]);
		Path classes_path = new Path("/task_2_1.classes");
		Path clusters_path = new Path("/task_2_1.clusters");
		FileSystem fs = FileSystem.get(conf);
		
	
		if (fs.exists(new Path(classes_path.toString()))) {
			fs.delete(new Path(classes_path.toString()), true);
		}
		if (fs.exists(new Path(clusters_path.toString()))) {
			fs.delete(new Path(clusters_path.toString()), true);
		}
		
		
		
		conf.set("K", K.toString());
		
		for (Integer i = 0; i < INTERATIONS; i++) {
			
			// Job 1: find the cluster id for each point
			List<Pair<Float, Float>> k_centroids = new ArrayList<>();

			if (i == 0) {
				for (Integer j = 0; j < K; j++) {
					Random r = new Random();
					Float x_random = 0.25f + (r.nextFloat() * 0.98f);
					Float y_random = 0.03f + (r.nextFloat() * 0.92f);
					k_centroids.add(new Pair<>(x_random, y_random));
				}
			}
			else {
				// Read centroids from HDFS file				
				BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(clusters_path.toString() + (i - 1) + "/part-r-00000")))); 
				String line;
			    while ((line = reader.readLine()) != null) {
			      String[] data = line.split("\t");
			      Float x = Float.valueOf(data[0]);
			      Float y = Float.valueOf(data[1]);
			      k_centroids.add(new Pair<>(x, y));
			    }
			    
			    if (k_centroids.size() != K) {
			    	k_centroids = new ArrayList<>();
			    	
			    	for (Integer j = 0; j < K; j++) {
						Random r = new Random();
						Float x_random = 0.25f + (r.nextFloat() * 0.98f);
						Float y_random = 0.03f + (r.nextFloat() * 0.92f);
						k_centroids.add(new Pair<>(x_random, y_random));
					}
			    }
			}
			
			// add centroids to global variables
			Integer index = 0;
			for (Pair<Float, Float> p: k_centroids) {
				conf.set("centroid" + index.toString(), p.getKey().toString() + "\t" + p.getValue().toString());
				index += 1;
			}
			
//			output_path = new Path(output_path.toString() + i);
			Job job = Job.getInstance(conf, "Task 2.1: K Means: find classes");
			job.setJarByClass(Task_2_1_main.class);
			job.setMapperClass(Task_2_1_map.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(IntWritable.class);		
			FileInputFormat.addInputPath(job, input_path);
			FileOutputFormat.setOutputPath(job, new Path(classes_path.toString() + i));
			
			job.waitForCompletion(true);
			
			// Job 2: get new centroids
//			output_path = new Path(output_path.toString() + i);
			Job job1 = Job.getInstance(conf, "Task 2.1: K Means: update new centroids");
			job1.setJarByClass(Task_2_1_main.class);
			job1.setMapperClass(Task_2_1_cluster_map.class);
			job1.setReducerClass(Task_2_1_cluster_reduce.class);
			job1.setMapOutputKeyClass(Text.class);
		    job1.setMapOutputValueClass(Text.class);
			job1.setOutputKeyClass(Text.class);
			job1.setOutputValueClass(FloatWritable.class);		
			FileInputFormat.addInputPath(job1, new Path(classes_path.toString() + i));
			FileOutputFormat.setOutputPath(job1, new Path(clusters_path.toString() + i));
			
//			input_path = output_path;
			job1.waitForCompletion(true);

		}
		
		
		fs.rename(new Path(classes_path.toString() + (INTERATIONS - 1) + "/part-r-00000"), new Path("/task_2_1.classes"));
		fs.rename(new Path(clusters_path.toString() + (INTERATIONS - 1) + "/part-r-00000"), new Path("/task_2_1.clusters"));
		
		for (Integer i = 0; i < K; i++) {
			if (fs.exists(new Path(classes_path.toString() + i))) {
				fs.delete(new Path(classes_path.toString() + i), true);
			}
			if (fs.exists(new Path(clusters_path.toString() + i))) {
				fs.delete(new Path(clusters_path.toString() + i), true);
			}
		}
		
		System.exit(0);
	}
}
