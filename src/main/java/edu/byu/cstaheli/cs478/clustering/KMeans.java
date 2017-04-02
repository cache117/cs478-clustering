package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.learner.UnsupervisedLearner;
import edu.byu.cstaheli.cs478.toolkit.utility.Matrix;
import org.jetbrains.annotations.NotNull;

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

    @NotNull
    private ArrayList<Cluster> populateInitialCentroids(Matrix dataset)
    {
        Map<Integer, Cluster> centroids = new HashMap<>(k);
        for (int i = 0; i < k; ++i)
        {
            int row = getNewRandomRow(dataset);
            double[] centroid = dataset.row(row);
            centroids.put(row, new Cluster(centroid, dataset));
        }
        return new ArrayList<>(centroids.values());
    }

    private int getNewRandomRow(Matrix dataset)
    {
        int row;
        do
        {
            row = getRandomRow(dataset.rows());
        } while (rowAlreadyChosen(row));
        return row;
    }

    @Override
    public void cluster(Matrix dataset)
    {
        clusters = populateInitialCentroids(dataset);
        boolean keepTraining;
        do
        {
            clearClusters();
            addRowsToClusters(dataset);
            calculateNewCentroids();
            keepTraining = shouldKeepTraining();
        } while (keepTraining);
    }

    private void clearClusters()
    {
        for (Cluster cluster : clusters)
        {
            cluster.clear();
        }
    }

    private void addRowsToClusters(Matrix dataset)
    {
        for (int i = 0; i < dataset.rows(); ++i)
        {
            double[] row = dataset.row(i);
            Cluster bestCluster = getBestClusterForRow(row);
            bestCluster.add(row);
        }
    }

    private Cluster getBestClusterForRow(double[] row)
    {
        double bestDistance = Double.MAX_VALUE;
        Cluster bestCluster = clusters.get(0);
        for (Cluster cluster : clusters)
        {
            double distance = cluster.calcDistance(row);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestCluster = cluster;
            }
        }
        return bestCluster;
    }

    private void calculateNewCentroids()
    {
        for (Cluster cluster : clusters)
        {
            cluster.calculateNewCentroid();
        }
    }

    private boolean shouldKeepTraining()
    {
        return false;
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

    private double silhouetteMetric(double ai, double bi)
    {
        if (Double.compare(ai, bi) == -1)
        {
            return 1 - (ai / bi);
        }
        else if (Double.compare(ai, bi) == 0)
        {
            return 0;
        }
        else
        {
            return (bi / ai) - 1;
        }
    }
}
