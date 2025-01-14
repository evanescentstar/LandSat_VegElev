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
    elevations within each pixel's boundaries - converted to CoNED's CRS - and applying the same upper vegetated
    fraction percentile (UVFP) filter to the elevations, and taking the mean in order to find the vegetated elevation.
    Then all pixels' vegetated elevations for that CMU were averaged together to yield a final CMU vegetated elevation
    estimate.  This should provide a way to compare vegetated elevations estimated from Landsat-based vegetated fractions
    to the published CMU values.
    
3. Phase_3_Chesa_LS_only.py - Script for phase 3 of the Chesapeake Landsat program.  This script estimates vegetated 
    elevations only on the individual Landsat pixels, without restricting results to CMU regions.  The output from the
    first two scripts were originally recorded into pandas dataframes, but here the results are recorded into an output
    GeoTIFF on the same grid as the Landsat pixel data (though a subset of the grid only in the Chesapeake region).
    
SAB - a folder containing scripts for estimating vegetated elevations and resultant marsh lifespans on the 10-m grid of 
    bias-analyzed elevations.  The data files of these elevations are not present in this repo.

