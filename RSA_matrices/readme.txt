data of representational similarity analysis from Castaldi et al 2019

the files named  RDM_allsub_task*_*.mat contain the representational dissimilarity matrices of all subjects for a given brain region tested in the fMRI study
- reading in such a file will result in a 3D matrix of dimension 18*18*20 (the last dimension corresponds to the subjects, these are labelled 4 to 23 in the stimuli, but the order is equivalent)

the file named all_RDM_predictors.mat contains the predictor matrices used in the study
- reading it in will result in a matrix of 18x18x5 (the last dimension corresponds to the individual predictors ordered: Number, Size, TFA, TSA, Density)

to create the fMRI RDMs:
- the evoked activity patterns across voxels (volumetric pixels of the fMRI image) within each region were averaged across all the presentations of each one of the 18 individual stimulus conditions
- then a scaling was performed for the activations of each voxel (subtracting from each voxel the mean of its activation across all the 18 conditions)
- finally, the correlation distance (1-Pearson correlation) was computed between the patterns of all 18 conditions, resulting in the 18x18 matrix

to perform the multiple regression on the RDMs which yields the beta weights for the different numerical dimensions:
- all the data (fMRI and predictor matrices) were first restricted to the lower triangular part of each matrix without the diagonal (this is important, since including the diagonal which is zero can lead to nonsensical results)
- all matrices (fMRI RDMs and predictors) were individually z-transformed
- then the multiple regression was computed on the fMRI data using all the predictor matrices together (yielding 5 beta weights for each brain region and subject)

NOTE: there are different files for taskNum, taskSize, and taskBoth, because subjects in the experiment had a task to either attend to number or average item size
- for the analysis of the model, it might make most sense to compare with the taskBoth (but we could also try taskNum)