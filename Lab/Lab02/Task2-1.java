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
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMean {

    public static class KMeansMapper extends Mapper<Object, Text, Text, IntWritable> {
        private List<Pair<Float, Float>> kCentroids = new ArrayList<>();
        private boolean firstLine = true;

        @Override
        protected void setup(Context context) {
            int k = Integer.parseInt(context.getConfiguration().get("K"));
            for (int i = 0; i < k; i++) {
                String centroid = context.getConfiguration().get("centroid" + i);
                String[] point = centroid.split("\t");
                kCentroids.add(new Pair<>(Float.valueOf(point[0]), Float.valueOf(point[1])));
            }
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            if (firstLine) {
                firstLine = false;
                return;
            }
            String[] data = value.toString().split(",");
            float x = Float.valueOf(data[1]);
            float y = Float.valueOf(data[2]);

            double minDistance = Double.MAX_VALUE;
            int clusterId = 0;
            int currentId = 0;
            for (Pair<Float, Float> centroid : kCentroids) {
                float cx = centroid.getKey();
                float cy = centroid.getValue();
                double distance = Math.sqrt(Math.pow(cx - x, 2) + Math.pow(cy - y, 2));
                if (distance < minDistance) {
                    minDistance = distance;
                    clusterId = currentId;
                }
                currentId++;
            }

            context.write(new Text(x + "\t" + y), new IntWritable(clusterId));
        }
    }

    public static class KMeansReducer extends Reducer<Text, Text, Text, FloatWritable> {

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            float xSum = 0.0f;
            float ySum = 0.0f;
            int total = 0;
            for (Text value : values) {
                String[] data = value.toString().split("\t");
                float x = Float.valueOf(data[0]);
                float y = Float.valueOf(data[1]);
                xSum += x;
                ySum += y;
                total++;
            }

            float xMean = xSum / total;
            float yMean = ySum / total;
            context.write(new Text(xMean + "\t" + yMean), new FloatWritable(yMean));
        }
    }

    public static class ClusterMapper extends Mapper<Object, Text, Text, Text> {

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] data = value.toString().split("\t");
            String x = data[0];
            String y = data[1];
            String id = data[2];
            context.write(new Text(id), new Text(x + "\t" + y));
        }
    }

    public static void main(String[] args) throws Exception {
        int K = 3;
        int ITERATIONS = 20;
        Configuration conf = new Configuration();
        Path inputPath = new Path(args[0]);
        Path classesPath = new Path("/task_2_1.classes");
        Path clustersPath = new Path("/task_2_1.clusters");
        FileSystem fs = FileSystem.get(conf);

        if (fs.exists(classesPath)) {
            fs.delete(classesPath, true);
        }
        if (fs.exists(clustersPath)) {
            fs.delete(clustersPath, true);
        }

        conf.set("K", String.valueOf(K));

        for (int i = 0; i < ITERATIONS; i++) {
            List<Pair<Float, Float>> kCentroids = new ArrayList<>();

            if (i == 0) {
                Random random = new Random();
                for (int j = 0; j < K; j++) {
                    float x = 0.25f + random.nextFloat() * 0.98f;
                    float y = 0.03f + random.nextFloat() * 0.92f;
                    kCentroids.add(new Pair<>(x, y));
                }
            } else {
                BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(clustersPath.toString() + (i - 1) + "/part-r-00000"))));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] data = line.split("\t");
                    float x = Float.valueOf(data[0]);
                    float y = Float.valueOf(data[1]);
                    kCentroids.add(new Pair<>(x, y));
                }
                reader.close();

                if (kCentroids.size() != K) {
                    kCentroids.clear();
                    Random random = new Random();
                    for (int j = 0; j < K; j++) {
                        float x = 0.25f + random.nextFloat() * 0.98f;
                        float y = 0.03f + random.nextFloat() * 0.92f;
                        kCentroids.add(new Pair<>(x, y));
                    }
                }
            }

            int index = 0;
            for (Pair<Float, Float> p : kCentroids) {
                conf.set("centroid" + index, p.getKey() + "\t" + p.getValue());
                index++;
            }

            Job job = Job.getInstance(conf, "KMeans: Find Classes");
            job.setJarByClass(KMean.class);
            job.setMapperClass(KMeansMapper.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);
            FileInputFormat.addInputPath(job, inputPath);
            FileOutputFormat.setOutputPath(job, new Path(classesPath.toString() + i));
            job.waitForCompletion(true);

            Job job1 = Job.getInstance(conf, "KMeans: Update Centroids");
            job1.setJarByClass(KMean.class);
            job1.setMapperClass(ClusterMapper.class);
            job1.setReducerClass(KMeansReducer.class);
            job1.setMapOutputKeyClass(Text.class);
            job1.setMapOutputValueClass(Text.class);
            job1.setOutputKeyClass(Text.class);
            job1.setOutputValueClass(FloatWritable.class);
            FileInputFormat.addInputPath(job1, new Path(classesPath.toString() + i));
            FileOutputFormat.setOutputPath(job1, new Path(clustersPath.toString() + i));
            job1.waitForCompletion(true);
        }

        fs.rename(new Path(classesPath.toString() + (ITERATIONS - 1) + "/part-r-00000"), new Path("/task_2_1.classes"));
        fs.rename(new Path(clustersPath.toString() + (ITERATIONS - 1) + "/part-r-00000"), new Path("/task_2_1.clusters"));

        for (int i = 0; i < K; i++) {
            if (fs.exists(new Path(classesPath.toString() + i))) {
                fs.delete(new Path(classesPath.toString() + i), true);
            }
            if (fs.exists(new Path(clustersPath.toString() + i))) {
                fs.delete(new Path(clustersPath.toString() + i), true);
            }
        }

        System.exit(0);
    }
}
