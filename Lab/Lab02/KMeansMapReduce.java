import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.naming.Context;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class KMeansMapReduce {
	public static final int _K 			= 3;
	public static final int _MAX_ITER 	= 20;
	//public static final String _CENTROIDS_GLOBAL 		= "centroids";		// key for conf.set() and conf.get()
	//public static final String _NEW_CENTROIDS_GLOBAL 	= "new_centroids";	// key for conf.set() and conf.get()

	public static Point[] _init_centroids;
	
	public static class Point {
		double[] _x;
		
		Point(int dim, double val) {
			this._x = new double[dim];
			for (int i = 0; i < dim; i++) {
				this._x[i] = val;
		}
	}
		
		Point(double[] x) {
			/* Contructor
			 * Copy x to this._x
			 */
			this._x = x;
		}

		Point(String str, String separator) {
			/* Contructor
			 * Parse a string to Point. Ex: "1.0,2.0,3.0"
			 */
			String[] tokens = str.split(separator);

			this._x = new double[tokens.length];
			for (int i = 0; i < tokens.length; i++) {
				this._x[i] = Double.parseDouble(tokens[i]);
			}
		}

		public double distance(Point p) {
			/*
			 * Euclidean distance of 2 Points
			 */
			double dis = 0;
			for (int i = 0; i < this._x.length; i++) {
				dis += (this._x[i] - p._x[i]) * (this._x[i] - p._x[i]);
			}
			return Math.sqrt(dis);
		}

		public int minDistanceIndex(Point[] points) {
			/*
			 * Find the index of the nearest point in the list
			 */
			int min_index = -1;
			double min_distance = Double.MAX_VALUE;
			for (int i = 0; i < points.length; i++) {
				double dis = distance(points[i]);
				if (dis < min_distance) {
					min_distance = dis;
					min_index = i;
				}
			}
			return min_index;
		}

		public String toString(String separator) {
			/*
			 * Convert object to String for debug, write to files
			 */
			String s = this._x[0] + "";
			for (int i = 1; i < this._x.length; i++)
				s += separator + this._x[i];
			return s;
		}

		public String toString() {
			/*
			 * Convert object to String in default format (1.11, 2.22)
			 */
			return "(" + toString(", ") + ")";
		}
		
		/*
		 * 		STATIC FUNCTIONS
		 */
		public static String toString(Point[] points) {
			/*
			* Point[] to String. ex [(1,1), (2,2), (3,3)] -> "1,1/2,2/3,3"
			*/
			String s = points[0].toString(",");
			
			for (int i = 1; i < points.length; i++)
			s += "/" + points[i].toString(",");
			
			return s;
		}
		
		public static Point[] parse(String s) {
			/*
			 * String to Point[]. ex "1,1/2,2/3,3" -> [(1,1), (2,2), (3,3)]
			 */
			String[] tokens = s.split("/");
			Point[] points = new Point[tokens.length];
		
			for (int i = 0; i < tokens.length; i++) {
				points[i] = new Point(tokens[i], ",");
			}
			return points;
		}
		
		public static Point centroid(Point[] list) {
			/*
			 * Find the centroid for a set of Points
			 */
			Point cent = new Point(list[0]._x.length, 0);
			
			for (Point p : list) {
				for (int i = 0; i < cent._x.length; i++) {
					cent._x[i] += p._x[i];
				}
			}
			
			for (int i = 0; i < cent._x.length; i++) {
				cent._x[i] /= list.length;
			}
			
			return cent;
		}
		
		public static Point[] randomPoints(int k, int dim) {
			/*
			* Random a list of Points length=k, in dim dimension
			 */
            Random random = new Random();
            Point[] centroids = new Point[k];
            
            for (int i = 0; i < k; i++) {
                double[] coordinates = new double[dim];
                for (int j = 0; j < dim; j++) {
                    coordinates[j] = random.nextDouble(); // Random value between 0.0 and 1.0
                }
                centroids[i] = new Point(coordinates);
            }
            return centroids;
        }
        
		public static boolean isSame(Point p1, Point p2) {
			/*
			 * 2 Points are the same if each |p1.x_i - p2.x_i| < eps.
			 */
		  double eps = 0.000001;
		  for (int i = 0; i < p1._x.length; i++)
		    if (Math.abs(p1._x[i] - p2._x[i]) > eps)
		      return false;
		  return true; 
		}

		public static boolean isSame(Point[] l1, Point[] l2) {
			/*
			 * 2 lists of Points are the same if each pair of Point of them are the same.
			 */
		  for (int i = 0; i < l1.length; i++) 
		    if (!isSame(l1[i], l2[i]))
		      return false;
		  return true;
		}
	}

	public static class KMeansMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			// Get the GLOBAL current centroids
			Point[] centroids = Point.parse(context.getConfiguration().get("centroids"));

			String line = value.toString().trim();	// Trim whitespace from the line
			if (!line.isEmpty()) {					// Check if the line is not empty
				Point p = new Point(line, "\t");
				int nearest_idx = p.minDistanceIndex(centroids);
				context.write(new IntWritable(nearest_idx), value);
			}
		}
	}

	public static class KMeansReducer extends Reducer<IntWritable, Text, NullWritable, Text> {
	
		protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			// Get all <key, list(values)>
			List<Point> points = new ArrayList<>();
			for (Text value : values) {
				points.add(new Point(value.toString(), "\t"));
			}
			
			// Calculate new centroid for this cluster
			Point new_centroid = Point.centroid(points.toArray(new Point[points.size()]));

			// Write the new centroid
			context.write(NullWritable.get(), new Text(new_centroid.toString("\t")));
		}
	}
	
	private static void copyAndRenameFile(FileSystem fs, Path src, Path dest) throws IOException {
       // Check if the source file exists
        if (fs.exists(src)) {
            // Copy the file to the destination
            boolean copied = fs.rename(src, dest);
        } else {
            System.out.println("Source file " + src + " does not exist.");
        }
    }

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		if (args.length != 2) {
			System.err.println("Usage: KMeansMapReduce <inputPath> <outputPath>");
			System.exit(1);
		}
		
		/*
		KMeans
		- input
			- input.csv
		- output
			- iter_1
				- clusters
					- part-r-00000
				- classes
					- part-m-00000
					...
			- iter_i
				- clusters
					- part-r-00000
				- classes
					- part-m-00000

			task_2_1.clusters
			task_2_1.classes
		*/
		
		// Init centroids [.69, .8],[1, .8],[.8, .6]
		KMeansMapReduce._init_centroids = new Point[] {new Point("0.6,0.8", ","), new Point("1,0.8", ","), new Point("0.8,0.6", ",")};

		int iter = 0;
		while (iter < KMeansMapReduce._MAX_ITER) {
			iter++;
			
			/*
			 *		JOB 1: ASSIGN THE CLASSES
			 */
			// Create a new job
			Configuration conf = new Configuration();
			
			// Push centroids to GLOBAL
			conf.set("centroids", Point.toString(KMeansMapReduce._init_centroids));
			Job job1 = Job.getInstance(conf, "Job 1: Assign the classes. Iter=" + iter);
			job1.setJarByClass(KMeansMapReduce.class);
			
			// Set input and output paths
			Path output_classes_path = new Path(args[1] + "/iter_" + iter + "/classes/");
			FileInputFormat.addInputPath(job1, new Path(args[0]));
			FileOutputFormat.setOutputPath(job1, output_classes_path);
			
			// Set Mapper class
			job1.setMapperClass(KMeansMapper.class);
			
			// Set output key and value classes
			job1.setOutputKeyClass(LongWritable.class);
			job1.setOutputValueClass(Text.class);
			
			// Set the number of reducers to zero
			job1.setNumReduceTasks(0);
			
			// Execute the job
			try {
				job1.waitForCompletion(true);
			} catch (Exception e) {
				// Handle job execution exceptions
				e.printStackTrace();
			}
			
			/*
			 *		JOB 2: FIND NEW CENTROIDS
			 */
			//Configuration conf = new Configuration();
  
			if (iter != 1) {
				Path previous_iter_path = new Path(args[1] + "/iter_" + (iter - 1) + "/clusters/part-r-00000");
				FileSystem fs = FileSystem.get(conf);
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(previous_iter_path)));
				
				// Read centroid file line by lines
				String line; int i = 0;
				while ((line = br.readLine()) != null) {
					_init_centroids[i++] = new Point(line, "\t");
				}
				br.close();
			}
  
			Job job2 = Job.getInstance(conf, "Job 2: Find new centroids. Iter=" + iter);
			
			job2.setJarByClass(KMeansMapReduce.class);
			job2.setMapperClass(KMeansMapper.class);
			job2.setReducerClass(KMeansReducer.class);
			  
			job2.setMapOutputKeyClass(IntWritable.class);
			job2.setMapOutputValueClass(Text.class);
			job2.setOutputKeyClass(NullWritable.class);
			job2.setOutputValueClass(Text.class);
			
			Path output_path = new Path(args[1] + "/iter_" + iter + "/clusters");
			FileInputFormat.addInputPath(job2, new Path(args[0]));
			FileOutputFormat.setOutputPath(job2, output_path);
			

			try {
				job2.waitForCompletion(true);
			} catch (Exception e) {
				// Handle job execution exceptions
				e.printStackTrace();
			}

			// Read new centroid file line by lines
			Path current_iter_path = new Path(args[1] + "/iter_" + iter + "/clusters/part-r-00000");
			FileSystem fs = FileSystem.get(conf);
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(current_iter_path)));
			Point[] new_centroids = KMeansMapReduce._init_centroids.clone();
			String line; int i = 0;
			while ((line = br.readLine()) != null) {
				_init_centroids[i++] = new Point(line, "\t");
			}
			br.close();
			
			// Stop condition
			if (Point.isSame(KMeansMapReduce._init_centroids, new_centroids) || iter >= KMeansMapReduce._MAX_ITER) {
				// Copy and rename files
				copyAndRenameFile(fs, new Path(args[1] + "/iter_" + iter + "/classes/part-m-00000"), new Path(args[1] + "/task_2_1.classes"));
				copyAndRenameFile(fs, new Path(args[1] + "/iter_" + iter + "/clusters/part-r-00000"), new Path(args[1] + "/task_2_1.clusters"));
				
				// Close FileSystem
				fs.close();
				break;
			}
		}
		
		System.exit(0);
	}
}
