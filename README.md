I've tried writing this readme several times but haven't been saved.

Currently, I've saved a single value result for visualization.

Based on this result, I'm considering performing a latitude and longitude determination on the output R@10, as it's easy to find the correct location at positions 234 and 259, while only looking at the first three or five positions is prone to error. Higher image quality is better when feeding the image into subsequent processes.

Model Calls After Training

The model iterates through gallery slices to generate pt feature vector files, then generates feature vectors for subsequent drone video frames for matching. However, my current structure is a dual-tower configuration, which presents a problem.

I'm performing latitude and longitude analysis on the corresponding R@10 candidate list, giving higher recommendations to results closer to each other. If the overall structure is very loose and the confidence level is low, should I trigger a gpt call?

How do I demonstrate the clustering characteristic?

The clustering characteristic essentially means continuous feature vectors in time and multiple feature vectors in space. Feature vectors represent the located clips.

Mobile RTK base stations imply that extremely high relative position estimation is reasonable and realistic, so let's assume this is the premise.

Let me outline my thinking. I have a model that can find the slice corresponding to the drone's downward-looking view from satellite slices.

The inference at time k for drone A (A) means a localization operation, recording the position of candidate R@10. Drone B performs the same inference at time k.

Given the known relative position difference and direction between the two, finding the constrained localization among the two R@10 candidates is more accurate than localization for a single drone.

Time-based localization also needs to be considered; the distance difference between the localization at time k+1 and time k is also a constraint. Time and space both need to be smoothed, and the confidence result sequence needs to be obtained. Gemini indicates that for the subsequent localization problem, 1. train another model to find the best match in the large slice to improve the system accuracy. 2. By optimizing the constraint method, the relative distance between k drones is obtained as the constraint condition, and the R@10 condition is used as the solution space, because its localization will not exceed the sqrt(2) of the slice. The slice size can be controlled, unlike the repeated sequence frames, which will mislead people. So now I need to reconstruct the dataset, including the satellite full map slice, latitude and longitude positions. If you want to save the calculation, you can replace all the latitude and longitude with the absolute distance coordinates in the satellite map in advance. Here I need to determine the model I use and use it to generate the gallery's pt. Now the drone's image is too big. 01 The first image is 390m width 250m height. But as long as it can be located within this slice, it is still acceptable? Using satellite image 02 for tiling, at 300x300 or 500 pixels, the 224 resolution completely loses detail.

Tomorrow we'll unify the data, including doing optimization problems, followed by visualization and sequence visualization. If possible, we'll consider directly tiling the satellite image as drone images, thus avoiding the constraints of aerial photography sequences.
