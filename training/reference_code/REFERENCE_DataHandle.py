import numpy as np
from collections import defaultdict


# List of NLCD classes. Some are specific to Alaska and so will not be seen at
# all in our data. I believe that "0" represents no data; I'm not sure why
# 255 is included -- it's possibly vestigal from another data export method.
# Note that I once saw an error due to a value of -128, so switched to
# defaultdict approach.
NLCD_CLASSES = [
    0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95, 255
]
NLCD_CLASS_NAMES = {
    0:  "NOT A CLASS",
    11: "Open water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Inensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren land (Rock/Sand/Clay)",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    51: "Dwarf Scrub (Alaska only)",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    72: "Sedge/Herbaceous (Alaska only)",
    73: "Lichens (Alaska only)",
    74: "Moss (Alaska only)",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
    255: "NOT A CLASS"
}
assert len(NLCD_CLASSES) == len(NLCD_CLASS_NAMES)

def get_nlcd_stats():
    '''Returns some values pre-computed by Kolya.

    cid:          dictionary mapping each NLCD label to a consecutive index.
    nlcd_if:      inverse frequencies of the NLCD labels in the Chesapeake Bay
                  (used for downsampling regions with common labels).
    nlcd_dist:    matrix defining the middle of the acceptable interval for each
                  NLCD-LC pair (used for interval constraints).
    nlcd_var:     matrix defining the radius of the acceptable interval for each
                  NLCD-LC pair (used for interval constraints).
    '''

    cid = defaultdict(lambda: 0, {cl:i for i,cl in enumerate(NLCD_CLASSES)})

    ## Load the counts of how frequently each category was observed, take the square
    ## root (why -- is this just dampening?), and indicate missingness
    #nlcd_f = np.sqrt(np.loadtxt('data/nlcd_counts_new.txt'))
    nlcd_f = np.sqrt(np.loadtxt('data/nlcd_counts_new.txt'))
    nlcd_f /= nlcd_f.sum()
    nlcd_f[nlcd_f == 0] = -1.

    # Calculate the inverse frequency. I'm not sure why we have to handle NaNs now
    # since we ensured nlcd_f != 0
    nlcd_if = 1 / nlcd_f
    nlcd_if[np.isnan(nlcd_if)] = -1.

    # Set inverse frequency to zero for unseen labels, and NLCD classes not in the
    # MRLC legend (0 and 255)
    nlcd_if[nlcd_if < 0] = 0
    nlcd_if[0] = 0
    nlcd_if[-1] = 0

    # Increase the inverse frequency (thus raising the probability with which
    # samples will be retained) for the following labels:
    # 11: Water
    # 21: Developed, open space
    # 22: Developed, low-intensity
    # 23: Developed, medium-intensity
    # 24: Developed, high-intensity
    #nlcd_if[1] *= 2
    #nlcd_if[3:7] *= 8 # In one of Kolya's scripts, this factor is 2 #CHANGED BACK TO 2

    # Change the range of inverse frequencies back to 0, 1
    nlcd_if /= nlcd_if.max()

    #print("Inverse frequencies:")
    #print("-"*40)
    #for class_index, class_val in enumerate(NLCD_CLASSES):
    #    print("%d\t%0.3f\t%s" % (class_val, nlcd_if[class_index], NLCD_CLASS_NAMES[class_val]))
    #print("-"*40)

    # Leave a placeholder column for the "no data" CC land cover label index, 0.
    nlcd_dist = np.zeros((len(NLCD_CLASSES), 5))
    nlcd_dist[:, 1:] = np.loadtxt('data/nlcd_mu.txt')

    # Further adjustments: ignore counts for label "255"; slightly raise all zero
    # values, then set the "nodata" fractions to zero again.
    nlcd_dist[nlcd_dist == 0] = 0.000001
    nlcd_dist[:, 0] = 0

    # Now we "adjust" so that the fraction of pixels classified as water is lower
    # for any pixels with NLCD label > 11; the fraction of tree LC in forests is
    # approx. 1.0; the distribution of impervious LC in developed land is higher
    # than truly calculated.
    # NB: will cause some values to leave the [0, 1] interval
    nlcd_dist[2:,1] -= 0.25
    nlcd_dist[3:7, 4] += 0.25

    # Replicate the "normalization" performed around line 160 of Kolya's training
    # script, in case it is necessary to achieve similar results.
    nlcd_dist = nlcd_dist / np.maximum(0, nlcd_dist).sum(axis=1, keepdims=True)

    # Load the sigmas and ensure no values are too low.
    nlcd_var = np.zeros_like(nlcd_dist)
    nlcd_var[:, 1:] = np.loadtxt('data/nlcd_sigma.txt')
    nlcd_var[nlcd_var < 0.0001] = 0.0001

    return cid, nlcd_if, nlcd_dist, nlcd_var

