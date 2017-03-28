package edu.byu.cstaheli.cs478.toolkit.utility;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by cstaheli on 3/9/2017.
 */
public class Utility
{
    public static double[] removeColumnFromRow(int column, double[] row)
    {
        List<Double> newRow = Arrays.stream(row)
                .boxed()
                .collect(Collectors.toList());
        newRow.remove(column);
        return newRow.stream()
                .mapToDouble(i -> i)
                .toArray();
    }

    public static double square(double value)
    {
        return value * value;
    }

    public static <K, V extends Comparable<? super V>> Map<K, V> sortMapByValue(Map<K, V> map)
    {
        return map.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue(Collections.reverseOrder()))
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1,
                        LinkedHashMap::new
                ));
    }

    public static double getOutputFromRow(double[] row)
    {
        return row[row.length - 1];
    }
}
