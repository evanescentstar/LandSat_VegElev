# Landsat to Vegetated Elevation and Marsh Lifespan Scripts

1. Phase_1_Chesa_CMU_only.py - Script for first phase of a program to test using Landsat pixel-based vegetated fraction
    to estimate vegetated elevation within a marsh.  For this phase, we just take the CMU polygon for each marsh unit, 
    select the co-located CoNED elevations from the online raster CoNED dataset (in its own CRS - converting the CMU to
    that), select the fraction of elevations that
    comprise the upper percentile - equal to the vegetated fraction - of elevations, and take the mean of these.  This 
    value is reported as the vegetated elevation estimate of the CMU for this approach.
    
2. Phase_2_Chesa_LS_on_CMU.py - Script for phase 2 of the Chesapeake Landsat program.  In this algorithm, we find the
    Landsat pixels which are co-located with the CMU, meaning that the centroid for the pixel lies within the boundary
    of the CMU (converted to the Landsat dataset's CRS).  These Landsat pixels are then used to find co-located CoNED 