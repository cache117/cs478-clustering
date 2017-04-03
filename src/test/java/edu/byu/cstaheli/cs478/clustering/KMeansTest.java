package edu.byu.cstaheli.cs478.clustering;

import edu.byu.cstaheli.cs478.toolkit.MLSystemManager;
import org.junit.jupiter.api.Test;

/**
 * Created by cstaheli on 4/2/2017.
 */
class KMeansTest
{
    private static String datasetsLocation = "src/test/resources/datasets/clustering/";

    @Test
    void cluster() throws Exception
    {
        String[] args;
        MLSystemManager manager = new MLSystemManager();
        args = ("-L kmeans -A " + datasetsLocation + "labor_data.arff -E cluster 3 -V").split(" ");
        manager.run(args);
    }

}