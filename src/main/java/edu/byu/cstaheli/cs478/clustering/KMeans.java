package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.learner.UnsupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;

import java.util.*;

/**
 * Created by cstaheli on 4/1/2017.
 */
public class KMeans extends UnsupervisedLearner
{
    private int k;
    private Random random;
    private List<Cluster> clusters;

    public KMeans(int k, Random random)
    {
        this.k = k;
        this.random = random;
    }

    private ArrayList<Cluster> populateCentroids(Matrix dataset)
    {
        Map<Integer, Cluster> centroids = new HashMap<>(k);
        for (int i = 0; i < k; ++i)
        {
            int row;
            while (true)
            {
                row = getRandomRow(dataset.rows());
                if (rowAlreadyChosen(row))
                {
                    break;
                }
            }
            double[] centroid = dataset.row(row);
            centroids.put(row, new Cluster(centroid, dataset));
        }
        return new ArrayList<>(centroids.values());
    }

    @Override
    public void train(Matrix dataset)
    {
        clusters = populateCentroids(dataset);
        for (int i = 0; i < dataset.rows(); ++i)
        {
            for (Cluster cluster : clusters)
            {
                double distance = cluster.calcDistance(dataset.row(i));
            }
        }
    }

    @Override
    public void predict(double[] row, double[] label)
    {

    }

    public int getK()
    {
        return k;
    }

    public void setK(int k)
    {
        this.k = k;
    }

    private boolean rowAlreadyChosen(int row)
    {
        Cluster cluster = this.clusters.get(row);
        return cluster == null;
    }

    private int getRandomRow(int rows)
    {
        return random.nextInt(rows);
    }
}
