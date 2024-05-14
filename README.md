<h1>Collaborative Filtering-based Recommendation System Leveraging PySpark </h1>

<p>
Collaborative Filtering is commonly used in recommender systems where the aim to fill in the missing entries of a user-item association matrix. Pyspark uses the <strong> Alterating Least Square </strong> algorithm to learn the latent factors. I used PySpark, a scala programming language that helps us to interface with Resilient Distributed Datasets (RDDs) in Apache Spark and Python Programming language. Pyspark comes with other external libraries like <strong>
PySparkSQl,MLlib and GraphFrames  </strong>
</p>

<h2>Dataset: </h2>

<p>
 
Collected by Cai-Nicolas Ziegler in a 4-week crawl (August / September 2004) from the Book-Crossing community with kind permission from Ron Hornbaker, CTO of Humankind Systems. Contains 278,858 users (anonymized but with demographic information) providing 1,149,780 ratings (explicit / implicit) about 271,379 books.
  
http://www2.informatik.uni-freiburg.de/~cziegler/BX/
  </p>
  
  <h2>Exploratory Data Analysis using PySpark </h2>
  
<h4>Displaying the top 10 entries of the dataset</h4>

![image](https://user-images.githubusercontent.com/57468338/120422281-eb1a8980-c335-11eb-9b59-dc09c8ff739e.png)

<h4>Rename Columns using 'withColumn' function</h4>

![image](https://user-images.githubusercontent.com/57468338/120422426-359c0600-c336-11eb-850e-e44ffc70b7c1.png)

  
<h4>Changing the Schema of the DataFrame to integer type </h4>

![image](https://user-images.githubusercontent.com/57468338/120422537-7d229200-c336-11eb-94a3-5047026d8e6c.png)

  <h4>Data description</h4>

![image](https://user-images.githubusercontent.com/57468338/120422745-eb675480-c336-11eb-98cc-5e5598166d5c.png)

 <h2>Alternating Least Square  </h2>
<p> 
 Alternating Least Square matrix factorisation attempts to estimate the ratings matrix R as the product of two lower-rank matrices, X and Y, i.e. X * Yt = R. Typically these approximations are called ‘factor’ matrices. The general approach is iterative. During each iteration, one of the factor matrices is held constant, while the other is solved for using least squares. The newly-solved factor matrix is then held constant while solving for the other factor matrix.
  
Spark allows users to set the coldStartStrategy parameter to “drop” in order to drop any rows in the DataFrame of predictions that contain NaN values. The evaluation metric will then be computed over the non-NaN data and will be valid. Usage of this parameter is illustrated in the example below.
In the below section we will instantiate an ALS model, run hyperparameter tuning, cross validation and fit the model.

We perform random splits as in Spark’s CrossValidator or TrainValidationSplit, it is actually very common to encounter users and/or items in the evaluation set that are not in the training set. By default, Spark assigns NaN predictions during ALSModel.transform when a user and/or item factor is not present in the model.We set cold start strategy to ‘drop’ to ensure we don’t get NaN evaluation metrics
</p>

![image](https://user-images.githubusercontent.com/57468338/120423832-32564980-c339-11eb-8cc5-9d63ae889bf9.png)

 <h2>Performance Metrics</h2>
 
 <p>
  For measuring the Performance of the trained model I used Precision, Recall, NGCG (Normalized Discounted Cumulative Gain) and Mean average precision. The results are as follows:
  <ul>
    <li><strong>Precision: </strong>0.3151639176523556   </li>
    <li><strong>Recall: </strong>0.8654186806365645     </li>
     <li><strong>NDCG: </strong>0.9508211383261144     </li>
     <li><strong>Mean average precision: </strong>0.8655157034780928     </li>


    

</ul>
  
  
  
  
  </p>
