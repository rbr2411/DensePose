# DensePose: 
**Dense Human Pose Estimation In The Wild**

## Notebooks

### DensePose
See[`DensePose.ipynb`](DensePose.ipynb)
The 1st code cell is used to import drive in a colab notebook. Next, we imported all dependencies anaconda(with linux), pytorch, etc which were required for implementation of the model. After that, we clone densepose repo from its official github repository using
!git clone -q --depth 1 $git_repo_url
By this, we can use all python functions required to run our model from densepose directly.
!python2 $project_name/detectron/tests/test_spatial_narrow_as_op.py
!python2 $project_name/detectron/tests/test_zero_even_op.py
Above two files are used to check whether we are using correct dependencies and the correct environment required to run the model or not. 
After all this comes the implementation part. Here we first call input image(here demo_im.jpg) and model specifications are given in file “DensePose_ResNet101_FPN_s1x-e2e.yaml” present in configs folder(pre-trained weights are present in R-101.pkl). Output images will be created and stored in the DensePoseData/infer_out folder. 
To visualize the results, we will first read input file and output files created in previous cell and after that we will use these three line of codes to plot desired output images:
plt.imshow( im[:,:,::-1] )
plt.contour( IUV[:,:,1]/256.,10, linewidths = 1 )
plt.contour( IUV[:,:,2]/256.,10, linewidths = 1 )

## Now, for running the notebooks mentioned below, please clone the github repository in your google drive.

See[`notebooks/DensePose_COCO_Visualize.ipynb`](notebooks/DensePose_COCO_Visualize.ipynb) 
In this notebook, we visualize the DensePose-COCO annotations on the image. In the First cell we will import all the necessary modules. We then select a random image from the coco dataset and load the annotations corresponding to it. Then GetDensePoseMasks(Polys) function to get dense pose masks from the decoded masks. Then in the next cell, input data is clipped to the valid range for imshow with RGB data. Finally in the last cell, points are visualised.


See [`notebooks/DensePose-COCO-on-SMPL.ipynb`](notebooks/DensePose-COCO-on-SMPL.ipynb) 
This document demonstrates the localization of collected points on the SMPL model. AtFirst, we install chumpy by !pip install chumpy due to some errors in importing Pickle Module. In the next cell, we have defined some functions to visualize the SMPL model vertices like smpl_view_set_axis_full_body(ax,azimuth=0), etc. For the sake of simplicity, we have a single densepose annotation "demo_dp_single_ann.pkl" for demonstration.
Below, we load the ann and find corresponding face index and barycentric coordinates, which allows us to localize the point on the 3D surface.


### Visualize DensePose-RCNN Results:

See [`notebooks/DensePose_RCNN_Visualize_Results.ipynb`](notebooks/DensePose_RCNN_Visualize_Results.ipynb) 
In this notebook, in the first cell we import all the necessary modules. Then, we Visualize the I, U and V images. Then, we visualize the isocontours of the UV Fields. 

### DensePose-RCNN Texture Transfer:

See [`notebooks/DensePose-RCNN-Texture-Transfer.ipynb`](notebooks/DensePose-RCNN-Texture-Transfer.ipynb) 
In this notebook we demonstrate how the estimated dense coordinates can be used to map a texture from the UV space to image pixels. For this purpose we provide an "atlas" texture space, which allows easy design of custom textures. We also provide a modified texture that is obtained from the SURREAL dataset, which allows replication of the qualitative results that we provide in the paper.

### We suggest you to go through all these notebooks.
