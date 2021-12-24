# Self-Driving Car Beta Testing Nanodegree

This is a template submission for the midterm second course in the Udacity Self-Driving Car Engineer Nanodegree Program : 3D Object Detection (Midterm).

## 3D Object detection

We have used the Waymo Open Dataset's real-world data and used 3d point cloud for lidar based object detection.

    Configuring the ranges channel to 8 bit and view the range /intensity image (ID_S1_EX1)
    Use the Open3D library to display the lidar point cloud on a 3d viewer and identifying 10 images from point cloud.(ID_S1_EX2)
    Create Birds Eye View perspective (BEV) of the point cloud,assign lidar intensity values to BEV,normalize the heightmap of each BEV (ID_S2_EX1,ID_S2_EX2,ID_S2_EX3)
    In addition to YOLO, use the repository and add parameters ,instantiate fpn resnet model(ID_S3_EX1)
    Convert BEV coordinates into pixel coordinates and convert model output to bounding box format (ID_S3_EX2)
    Compute intersection over union, assign detected objects to label if IOU exceeds threshold (ID_S4_EX1)
    Compute false positives and false negatives, precision and recall(ID_S4_EX2,ID_S4_EX3)

The project can be run by running

``` 
python loop_over_dataset.py
```
All training/inference is done on GTX 2060 in windows 10 machine.

## Step-1: Compute Lidar point cloud from Range Image

In this we are first previewing the range image and convert range and intensity channels to 8 bit format. After that, we use the openCV library to stack the range and intensity channel vertically to visualize the image.

    Convert "range" channel to 8 bit
    Convert "intensity" channel to 8 bit
    Crop range image to +/- 90 degrees left and right of forward facing x axis
    Stack up range and intensity channels vertically in openCV

The changes are made in 'loop_over_dataset.py'
<img src="img/1.png" alt="img1"/>
<img src="img/2.png" alt="img2"/>

The changes are made in "objdet_pcl.py"

<img src="img/show_range_image_func.png" alt="img3"/>
The range image sample:
<img src="img/range_image.png" alt="range_image"/>

For the next part, we use the Open3D library to display the lidar point cloud on a 3D viewer and identify 10 images from point cloud

    Visualize the point cloud in Open3D
    10 examples from point cloud with varying degrees of visibility

The changes are made in 'loop_over_dataset.py' </br>
<img src="img/12.png" alt="img12"/>


The changes are made in "objdet_pcl.py" 
<img src="img/show_pcl_func.png" alt="imgpcl"/>

Point cloud images

<img src="img/pcl2.png" alt="imgpcl"/>

<img src="img/5.png" alt="img5"/>

<img src="img/13.png" alt="img13"/>

<img src="img/6.png" alt="img6"/>

<img src="img/9.png" alt="img9"/>

<img src="img/8.png" alt="img8"/>

<img src="img/10.png" alt="img10"/>

<img src="img/11.png" alt="img11"/>

Stable features include the tail lights, the rear bumper majorly. In some cases the additional features include the headover lights, car front lights, rear window shields. These are identified through the intensity channels . The chassis of the car is the most prominent identifiable feature from the lidar perspective. The images are analysed with different settings and the rear lights are the major stable components, also the bounding boxes are correctly assigned to the cars (used from Step-3).

## Step-2: Creaate BEV from Lidar PCL

In this case, we are:

    Converting the coordinates to pixel values
    Assigning lidar intensity values to the birds eye view BEV mapping
    Using sorted and pruned point cloud lidar from the previous task
    Normalizing the height map in the BEV
    Compute and map the intensity values

The changes are in the 'loop_over_dataset.py' </br>
<img src="img/14.png" alt="img14"/>

The changes are also in the "objdet_pcl.py"
<img src="img/15.png" alt="img15"/>

A sample preview of the BEV: </br>

<img src="img/bev_map.png" alt="imgbev"/>

## Step-3: Model Based Object Detection in BEV Image

Here we are using the cloned repo ,particularly the test.py file and extracting the relevant configurations from 'parse_test_configs()' and added them in the 'load_configs_model' config structure.

    Instantiating the fpn resnet model from the cloned repository configs
    Extracting 3d bounding boxes from the responses
    Transforming the pixel to vehicle coordinates
    Model output tuned to the bounding box format [class-id, x, y, z, h, w, l, yaw]

The changes are in "loop_over_dataset.py" </br>

<img src="img/16.png" alt="img16"/>

The changes for the detection are inside the "objdet_detect.py" file:

<img src="img/17.png" alt="img17"/>

As the model input is a three-channel BEV map, the detected objects will be returned with coordinates and properties in the BEV coordinate space. Thus, before the detections can move along in the processing pipeline, they need to be converted into metric coordinates in vehicle space.

A sample preview of the bounding box images:  </br>

<img src="img/18.png" alt="img18"/>

<img src="img/19.png" alt="img19"/>

## Step-4: Performance detection for 3D Object Detection

In this step, the performance is computed by getting the IOU between labels and detections to get the false positive and false negative values.The task is to compute the geometric overlap between the bounding boxes of labels and the detected objects:

    Assigning a detected object to a label if IOU exceeds threshold
    Computing the degree of geometric overlap
    For multiple matches objects/detections pair with maximum IOU are kept
    Computing the false negative and false positive values
    Computing precision and recall over the false positive and false negative values

The changes in the code are:

<img src="img/20.png" alt="img20"/>

The changes for "objdet_eval.py" where the precision and recall are calculated as functions of false positives and negatives:

<img src="img/22.png" alt="img22"/>

<img src="img/21.png" alt="img21"/>
