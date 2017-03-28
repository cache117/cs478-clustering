package edu.byu.cstaheli.cs478.nearest_neighbor;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

import static edu.byu.cstaheli.cs478.toolkit.utility.Utility.*;

/**
 * Created by cstaheli on 3/18/2017.
 */
public class Neighbors
{
    private int size;
    private Map<double[], Double> neighbors;
    private boolean useDistanceWeighting;
    private boolean useRegression;

    public Neighbors(int size, boolean useDistanceWeighting, boolean useRegression)
    {
        this.size = size;
        neighbors = new LinkedHashMap<>(size);
        this.useDistanceWeighting = useDistanceWeighting;
        this.useRegression = useRegression;
    }

    public void addPotential(double[] row, double distance)
    {
        boolean add = false;
        if (neighbors.size() < size)
        {
            add = true;
        }
        else
        {
            Iterator<Map.Entry<double[], Double>> iterator = neighbors.entrySet()
                    .iterator();
            while (iterator.hasNext())
            {
                Map.Entry<double[], Double> entry = iterator.next();
                if (Double.compare(entry.getValue(), distance) == 1)
                {
                    iterator.remove();
                    add = true;
                    break;
                }
            }
        }
        if (add)
        {
            neighbors.put(row, distance);
        }
        //Ensures that the entry with the largest distance is seen first and hence replaced first
        neighbors = sortMapByValue(neighbors);
    }

    public double predict()
    {
        if (useRegression)
        {
            return averageValue();
        }
        else
        {
            return mostCommonOutput();
        }
    }

    public double averageValue()
    {
        double sum = 0;
        if (useDistanceWeighting)
        {
            int timesToAdd = 0;
            for (Map.Entry<double[], Double> entry : neighbors.entrySet())
            {
                int weightingFactor = (int) (1 / entry.getValue()) + 1;
                timesToAdd += weightingFactor;
                for (int i = 0; i < weightingFactor; ++i)
                {
                    sum += getOutputFromRow(entry.getKey());
                }
            }
            return sum / (double) timesToAdd;
        }
        else
        {
            for (Map.Entry<double[], Double> entry : neighbors.entrySet())
            {
                sum += getOutputFromRow(entry.getKey());
            }
            return sum / (double) size;
        }
    }

    public double mostCommonOutput()
    {
        Map<Double, Integer> occurrences = getOccurrences();
        int maxCount = 0;
        double val = -1;
        for (Map.Entry<Double, Integer> entry : occurrences.entrySet())
        {
            if (entry.getValue() > maxCount)
            {
                maxCount = entry.getValue();
                val = entry.getKey();
            }
        }
        return val;
    }

    private Map<Double, Integer> getOccurrences()
    {
        Map<Double, Integer> occurrences = new TreeMap<>();
        for (Map.Entry<double[], Double> entry : neighbors.entrySet())
        {
            double output = getOutputFromRow(entry.getKey());
            Integer count = occurrences.get(output);
            if (count == null)
            {
                if (useDistanceWeighting)
                {
                    occurrences.put(output, (int) (1 / entry.getValue()) + 1);
                }
                else
                {
                    occurrences.put(output, 1);
                }
            }
            else
            {
                if (useDistanceWeighting)
                {
                    occurrences.put(output, count + (int) (1 / entry.getValue()) + 1);
                }
                else
                {
                    occurrences.put(output, count + 1);
                }
            }
        }
        return occurrences;
    }

    private double getDistanceWeight(double distance)
    {
        return 1d / square(distance);
    }

    public boolean isUseDistanceWeighting()
    {
        return useDistanceWeighting;
    }

    public void setUseDistanceWeighting(boolean useDistanceWeighting)
    {
        this.useDistanceWeighting = useDistanceWeighting;
    }

    public void setUseRegression(boolean useRegression)
    {
        this.useRegression = useRegression;
    }
}
