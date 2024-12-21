package com.projectgurukul.wc;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.apache.commons.math3.util.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Task_2_3_main {
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
    	
		Map<String, List<Pair<String, Float>>> res = new HashMap<String, List<Pair<String, Float>>>();;
		res.put(docId, pairs);
    	return res;
    }
	
	public static void main(String[] args) throws Exception {
		Integer K = 5;
		Integer INTERATIONS = 10;
		Configuration conf = new Configuration();
		Path input_path = new Path(args[0]);
//		Path output_path = new Path(args[1]);
		Path classes_path = new Path("/task_2_3.classes");
		Path clusters_path = new Path("/task_2_3.clusters");
		Path preprocessed_dir = new Path("/task_2_3");
		FileSystem fs = FileSystem.get(conf);
		
	
		if (fs.exists(new Path(classes_path.toString()))) {
			fs.delete(new Path(classes_path.toString()), true);
		}
		if (fs.exists(new Path(clusters_path.toString()))) {
			fs.delete(new Path(clusters_path.toString()), true);
		}
		if (fs.exists(new Path(preprocessed_dir.toString()))) {
			fs.delete(new Path(preprocessed_dir.toString()), true);
		}
		if (fs.exists(new Path("/task_2_3.loss"))) {
			fs.delete(new Path("/task_2_3.loss"), true);
		}
		
		if (fs.exists(new Path("/min_distance"))) {
			fs.delete(new Path("/min_distance"), true);
		}
		
		for (Integer i = 0; i < INTERATIONS; i++) {
			if (fs.exists(new Path(classes_path.toString() + i))) {
				fs.delete(new Path(classes_path.toString() + i), true);
			}
			if (fs.exists(new Path(clusters_path.toString() + i))) {
				fs.delete(new Path(clusters_path.toString() + i), true);
			}
			
			if (fs.exists(new Path("/task_2_3.loss" + i))) {
				fs.delete(new Path("/task_2_3.loss" + i), true);
			}
			
		}
		
		
		conf.set("K", K.toString());
		
		// input: termId - DocId - TFIDF, output: DocId termId1:TFIDF1, termId2:TFIDF2,...termIdn:TFIDFn
		Job job = Job.getInstance(conf, "Task 2.3: preprocessed");
		job.setJarByClass(Task_2_2_main.class);
		job.setMapperClass(Task_2_2_map1.class);
		job.setReducerClass(Task_2_2_reduce1.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);		
		FileInputFormat.addInputPath(job, input_path);
		FileOutputFormat.setOutputPath(job, preprocessed_dir);
		
		job.waitForCompletion(true);
		
		// Read all doc-id term-id tfidf
		BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(preprocessed_dir.toString() + "/part-r-00000")))); 
		String line;
		
		Map<String, String> docId_termId_tfidf = new HashMap<String, String>();
		ArrayList<String> docIds = new ArrayList<>();
		
	    while ((line = reader.readLine()) != null) {
	    	String[] data = line.split("\t");
	    	String docId = data[0];
	    	String termId_tfidf = data[1];
	    	docIds.add(docId);
	    	docId_termId_tfidf.put(docId,  termId_tfidf);
	    }
	    
        ArrayList<String> centroids = new ArrayList(); // store multiple centrois as a single string, each centroid seperated by "\n"
        
		for (Integer i = 0; i < INTERATIONS; i++) {
			System.out.println("----------------------------- Iteration " + i.toString());
			
			// get random k centroids
			if (centroids.size() != K) {
				
				Integer l = 2 * K;
				
				// random 1 centroid from X
				String[] newDocIds = docIds.toArray(new String[0]);
			    int random_int = (int)Math.floor(Math.random() * (newDocIds.length - 1));
			    String docId = newDocIds[random_int];
			    String c = docId_termId_tfidf.get(docId);
			    centroids.add(docId + "\t" + c);
			    
			    // write centrois to global variables
			    for (Integer index = 0; index < centroids.size(); index++) {
				    conf.set("centroid" + index, centroids.get(index));
			    } 
			    
			    // get the distance of 1 centroid to all of X
			    Job init_job = Job.getInstance(conf, "Task 2.3: K Means: class");
			    init_job.setJarByClass(Task_2_3_main.class);
			    init_job.setMapperClass(Task_2_3_map.class);
			    init_job.setOutputKeyClass(Text.class);
			    init_job.setOutputValueClass(FloatWritable.class);		
				FileInputFormat.addInputPath(init_job, new Path(preprocessed_dir.toString() + "/part-r-00000"));
				FileOutputFormat.setOutputPath(init_job, new Path("/min_distance"));
				 
				init_job.waitForCompletion(true);
				
				
				// read min_dis of 1 centroid
				BufferedReader min_dis_reader = new BufferedReader(new InputStreamReader(fs.open(new Path("/min_distance" + "/part-r-00000")))); 
				String min_dis_line;
				
				min_dis_line = min_dis_reader.readLine();
				String[] data = min_dis_line.split("\t");
			    Float min_dis = Float.valueOf(data[1]);
			    conf.set("min_dis", min_dis.toString());
			    conf.set("l", l.toString());
			    min_dis_reader.close();
				// loop log(min_dis) times
			    Integer cur_centroids = 1;
				for (Integer iter = 0; iter < Math.log(min_dis); iter++) {
					if (fs.exists(new Path("/probs" + iter))) {
						fs.delete(new Path("/probs" + iter), true);
					}
					
					 // get the distance of 1 centroid to all of X
				    conf.set("cur_centroids", cur_centroids.toString());
				    
					Job prob_job = Job.getInstance(conf, "Task 2.3: K Means: class");
				    prob_job.setJarByClass(Task_2_3_main.class);
				    prob_job.setMapperClass(Task_2_3_map2.class);
				    prob_job.setReducerClass(Task_2_3_reduce2.class);
				    prob_job.setMapOutputKeyClass(Text.class);
				    prob_job.setMapOutputValueClass(FloatWritable.class);
				    prob_job.setOutputKeyClass(Text.class);
				    prob_job.setOutputValueClass(FloatWritable.class);		
					FileInputFormat.addInputPath(prob_job, new Path(preprocessed_dir.toString() + "/part-r-00000"));
					FileOutputFormat.setOutputPath(prob_job, new Path("/probs" + iter.toString()));
					
					prob_job.waitForCompletion(true);
										
					//get id of new centroid
					
					BufferedReader new_c_reader = new BufferedReader(new InputStreamReader(fs.open(new Path(preprocessed_dir.toString() + "/part-r-00000")))); 
					String new_c_line;
					
				    while ((new_c_line = new_c_reader.readLine()) != null) {
				    	String[] dataa = new_c_line.split("\t");
				    	String new_docId = dataa[0];
				    	
				    	// get t-tfidf
				    	String new_t_tfidf = docId_termId_tfidf.get(new_docId);

						// add new l centroids to current centroids
					    centroids.add(new_docId + "\t" + new_t_tfidf);
				    }
				    
				    new_c_reader.close();
				    
				    // write to global variables
				    for (Integer index = 0; index < centroids.size(); index++) {
					    conf.set("centroid" + index, centroids.get(index));
				    } 
					
					cur_centroids += l;
					
				}
				
				// get random k centroids from l * log(min_dis) centroids in C 
				for (int o = 0; o < K; o++) {
					
				      int r_i = (int)Math.floor(Math.random() * (centroids.size() - 1));
				      
				      String r_c = centroids.get(r_i);
				      
//				      System.out.println(docId + "\t" + docId_termId_tfidf.get(docId));
				      conf.set("docId_termId_tfidf" + o, r_c);
				} 
				
			}
			else {

				Integer o = 0;
				System.out.println("We got new centroids");
				for (String centroid: centroids) {
					
					String termId_tfidf = centroid.split("\t")[1];
				    conf.set("docId_termId_tfidf" + o, o.toString() + "\t" + termId_tfidf);
					System.out.println(termId_tfidf);
				    o += 1;
				}
				System.out.println("\n");
			}
		    		
			// Job 2: find class 

			Job job1 = Job.getInstance(conf, "Task 2.3: K Means: class");
			job1.setJarByClass(Task_2_3_main.class);
			job1.setMapperClass(Task_2_2_map2.class);
			job1.setOutputKeyClass(Text.class);
			job1.setOutputValueClass(IntWritable.class);		
			FileInputFormat.addInputPath(job1, new Path(preprocessed_dir.toString() + "/part-r-00000"));
			FileOutputFormat.setOutputPath(job1, new Path(classes_path.toString() + i));
			 
			job1.waitForCompletion(true);

			
			// Job 3: find new centroids
			
			Job job2 = Job.getInstance(conf, "Task 2.3: K Means: find new centroids");
			job2.setJarByClass(Task_2_3_main.class);
			job2.setMapperClass(Task_2_2_map3.class);
			job2.setReducerClass(Task_2_2_reduce3.class);
			job2.setOutputKeyClass(Text.class);
			job2.setOutputValueClass(Text.class);		
			FileInputFormat.addInputPath(job2, new Path(classes_path.toString() + i + "/part-r-00000"));
			FileOutputFormat.setOutputPath(job2, new Path(clusters_path.toString() + i));
			
			job2.waitForCompletion(true);
			
			// read new centroids
			BufferedReader reader_centroid = new BufferedReader(new InputStreamReader(fs.open(new Path(clusters_path.toString() + i + "/part-r-00000")))); 
			String line1;
			
			centroids = new ArrayList<String>();
			
			// update new centroids 
		    while ((line1 = reader_centroid.readLine()) != null) {
		    	System.out.println(line1);
		    	
		    	centroids.add(line1);
		    }
		    
		    
		    // -----------  Top 10 important terms in each clusters ------------
		    Path FileToWrite = new Path("/task_2_3.txt" + i.toString()); 
	        FSDataOutputStream fileOut = fs.create(FileToWrite);
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fileOut));
			
		    for (String centroid: centroids) {
		    	TreeMap<Float, HashSet<String>> top10 = new TreeMap<>(Collections.reverseOrder());
		    	
		    	String[] data = centroid.split("\t");
		    	String docId = data[0];
			    
		    	writer.write(docId + ":\t");

		    	String termId_tfidf_list = data[1];
		    	
		    	String[] items = termId_tfidf_list.split(",");
		    	
		    	for (String item: items) {
		    		String[] tmp = item.split(":");
		    		String termId = tmp[0];
		    		Float tfidf = Float.valueOf(tmp[1]);
		    		
		    		if (! top10.containsKey(tfidf)) {
		    			top10.put(tfidf, new HashSet<>());
		    		}
		    		top10.get(tfidf).add(termId);
		    	}
		    	
				Integer count = 0;
				for (Map.Entry<Float, HashSet<String>> entry: top10.entrySet()) {
					for (String item: entry.getValue()) {
					    writer.write(item + "(" + entry.getKey().toString() +")\t");
						
					    count += 1;
					    if (count == 10) {
					    	break;
					    }
					}
					if (count == 10) break;
				}
				writer.write("\n");
		    }
		    writer.close();
	        fileOut.close();
		    
	        // turn all centroids to a string in order to make it global and be used in Loss function 
		    String centroids_for_loss = "";
		    for (String centroid: centroids) {
		    	if (centroids_for_loss == "") {
		    		centroids_for_loss = centroid;
		    	} else {
		    		centroids_for_loss += "\n" + centroid;
		    	}
		    }
		    
		    conf.set("centroid_loss", centroids_for_loss);
		    
		    // Job 4: loss function
		    Job job3 = Job.getInstance(conf, "Task 2.3: K Means: loss");
			job3.setJarByClass(Task_2_3_main.class);
			job3.setMapperClass(Task_2_2_map4.class);
			job3.setReducerClass(Task_2_2_reduce4.class);
			job3.setMapOutputKeyClass(Text.class);
			job3.setMapOutputValueClass(Text.class);
			job3.setOutputKeyClass(Text.class);
			job3.setOutputValueClass(FloatWritable.class);		
			FileInputFormat.addInputPath(job3, new Path(classes_path.toString() + i + "/part-r-00000"));
			FileOutputFormat.setOutputPath(job3, new Path("/task_2_3.loss" + i));
			
			job3.waitForCompletion(true);
		    
		}
		
		fs.rename(new Path(classes_path.toString() + (INTERATIONS - 1) + "/part-r-00000"), new Path("/task_2_3.classes"));
		fs.rename(new Path(clusters_path.toString() + (INTERATIONS - 1) + "/part-r-00000"), new Path("/task_2_3.clusters"));
		
		// Get loss from all files and write into one file "task_2_2.loss"
		Path FileToWrite = new Path("/task_2_3.loss"); 
        FSDataOutputStream fileOut = fs.create(FileToWrite);
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fileOut));
		for (Integer i = 0; i < INTERATIONS; i++) {
			// Read all doc-id term-id tfidf
			BufferedReader loss_reader = new BufferedReader(new InputStreamReader(fs.open(new Path("/task_2_3.loss" + i.toString() + "/part-r-00000")))); 
			String loss_line;
			
		    loss_line = loss_reader.readLine();
	    	writer.write(loss_line + "\n");

		    loss_reader.close();
		}
		writer.close();
	    fileOut.close();
		
		// Get top10 from all files and write into one file "task_2_2.txt"
		Path output = new Path("/task_2_3.txt"); 
        FSDataOutputStream fs_output = fs.create(output);
		BufferedWriter top10_writer = new BufferedWriter(new OutputStreamWriter(fs_output));
		for (Integer i = 0; i < INTERATIONS; i++) {
			// Read all doc-id term-id tfidf
			BufferedReader top10_reader = new BufferedReader(new InputStreamReader(fs.open(new Path("/task_2_3.txt" + i.toString())))); 
			String top10_line;
			
			while ((top10_line = top10_reader.readLine()) != null) {
				top10_writer.write(top10_line + "\n");
		    }

			top10_reader.close();
		}
		top10_writer.close();
		fs_output.close();
		
		System.exit(0);
	}
}
